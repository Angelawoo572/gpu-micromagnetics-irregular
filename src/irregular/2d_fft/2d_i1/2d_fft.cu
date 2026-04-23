/**
 * 2D periodic head-on transition LLG solver
 * CVODE + CUDA, SoA layout
 *
 * Anisotropy: x-axis  (c_msk={1,0,0}, c_nsk={1,0,0})
 *
 * Initial condition — professor's suggestion:
 *   mz = 0  (exactly, always)
 *   my = sin(phi(x))
 *   mx = sqrt(1 - my^2) = cos(phi(x))     [both signs handled via phi]
 *
 * phi(x) = pi * (1 - 0.5*(tanh((x-x1)/w) - tanh((x-x2)/w)))
 *
 *   x << x1:       phi → pi      →  mx=cos(pi)=-1,  my=sin(pi)=0   ✓
 *   x1 < x < x2:   phi → 0       →  mx=cos(0)=+1,   my=sin(0)=0    ✓
 *   x >> x2:       phi → pi      →  mx=-1,           my=0           ✓
 *   at x = x1:     phi = pi/2    →  mx=0,            my=+1  (Neel wall peak)
 *   at x = x2:     phi = pi/2    →  mx=0,            my=+1
 *
 * This is an in-plane Neel wall: the magnetization rotates continuously
 * through the +y direction at each domain wall.  |m|=1 is exact.
 *
 * Physical note on mz during dynamics:
 *   Even with mz(t=0)=0, the precession term
 *     dm3/dt = chg*(m2*h1 - m1*h2) + alpha*(h3 - mh*m3)
 *   is nonzero during transient (spins precess out of plane).
 *   This is physically correct — mz grows transiently and decays
 *   back to 0 as the system reaches equilibrium.
 *   The mz≠0 seen in results is transient behavior, not a bug.
 */

#include <cvode/cvode.h>
#include <cuda_runtime.h>
#include <math.h>
#include <nvector/nvector_cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>
#include <sundials/sundials_iterative.h>

#include "deferred_nvector.h"
#include "precond.h"
#include "jtv.h"
#include "demag_fft.h"

#define GROUPSIZE 3

#ifndef KRYLOV_DIM
#define KRYLOV_DIM 0
#endif

#ifndef MAX_BDF_ORDER
#define MAX_BDF_ORDER 5
#endif

#ifndef RTOL_VAL
#define RTOL_VAL 1.0e-5
#endif

#ifndef ATOL_VAL
#define ATOL_VAL 1.0e-5
#endif

#ifndef DEMAG_STRENGTH
#define DEMAG_STRENGTH 0.0
#endif

#ifndef DEMAG_THICK
#define DEMAG_THICK 1.0
#endif

#define RTOL  SUN_RCONST(RTOL_VAL)
#define ATOL1 SUN_RCONST(ATOL_VAL)
#define ATOL2 SUN_RCONST(ATOL_VAL)
#define ATOL3 SUN_RCONST(ATOL_VAL)

#define T0   SUN_RCONST(0.0)
#define T1   SUN_RCONST(0.1)
#define ZERO SUN_RCONST(0.0)

#ifndef T_TOTAL
#define T_TOTAL 1000.0
#endif

#ifndef ENABLE_OUTPUT
#define ENABLE_OUTPUT 0
#endif

#ifndef BLOCK_X
#define BLOCK_X 16
#endif

#ifndef BLOCK_Y
#define BLOCK_Y 8
#endif

/* Domain wall transition width as fraction of ng */
#ifndef WALL_WIDTH_FRAC
#define WALL_WIDTH_FRAC 0.05
#endif

#ifndef EARLY_SAVE_UNTIL
#define EARLY_SAVE_UNTIL 80.0
#endif

#ifndef EARLY_SAVE_EVERY
#define EARLY_SAVE_EVERY 5
#endif

#ifndef LATE_SAVE_EVERY
#define LATE_SAVE_EVERY 100
#endif

/*
 * Physical constants — x-axis easy axis
 *
 * c_msk = {1,0,0}:  H_ani_i = c_msk[i] * (c_chk * m1 + c_cha)
 *   → only h1 gets the anisotropy term: h1 += c_chk*m1 + c_cha
 *   → h2, h3 have no anisotropy
 *
 * c_nsk = {1,0,0}:  chb term only on h1, from x-neighbors of mx
 *   → h1 += c_chb * (mx_left + mx_right)  [anisotropic exchange]
 *   → h2, h3 unaffected by chb
 *
 * Resulting effective field (full expansion):
 *   h1 = (c_che+c_chb)*(mx_L+mx_R) + c_che*(mx_U+mx_D) + c_chk*m1 + c_cha
 *   h2 = c_che*(my_L+my_R+my_U+my_D)
 *   h3 = c_che*(mz_L+mz_R+mz_U+mz_D)
 */
__constant__ sunrealtype c_msk[3] = {
    SUN_RCONST(1.0), SUN_RCONST(0.0), SUN_RCONST(0.0)};
__constant__ sunrealtype c_nsk[3] = {
    SUN_RCONST(1.0), SUN_RCONST(0.0), SUN_RCONST(0.0)};

__constant__ sunrealtype c_chk   = SUN_RCONST(1.0);
__constant__ sunrealtype c_che   = SUN_RCONST(4.0);
__constant__ sunrealtype c_alpha = SUN_RCONST(0.2);
__constant__ sunrealtype c_chg   = SUN_RCONST(1.0);
__constant__ sunrealtype c_cha   = SUN_RCONST(0.0);
__constant__ sunrealtype c_chb   = SUN_RCONST(0.3);

#define CHECK_CUDA(call) \
  do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
    exit(EXIT_FAILURE); } } while(0)

#define CHECK_SUNDIALS(call) \
  do { int _f=(call); if(_f<0){ \
    fprintf(stderr,"SUNDIALS error %s:%d: flag=%d\n",__FILE__,__LINE__,_f); \
    exit(EXIT_FAILURE); } } while(0)

typedef struct {
  PrecondData  *pd;       /* offset 0  — must be first */
  DemagData    *demag;    /* offset 8  */
  sunrealtype  *d_hdmag;  /* offset 16 */
  int nx;                 /* offset 24 */
  int ny;                 /* offset 28 */
  int ng;                 /* offset 32 */
  int ncell;              /* offset 36 */
  int neq;                /* offset 40 */
} UserData;

__host__ __device__ static inline int idx_mx(int cell, int ncell) { return cell; }
__host__ __device__ static inline int idx_my(int cell, int ncell) { return ncell+cell; }
__host__ __device__ static inline int idx_mz(int cell, int ncell) { return 2*ncell+cell; }

__host__ __device__ static inline int wrap_x(int x, int ng) {
  return (x<0)?(x+ng):((x>=ng)?(x-ng):x);
}
__host__ __device__ static inline int wrap_y(int y, int ny) {
  return (y<0)?(y+ny):((y>=ny)?(y-ny):y);
}

/*
 * RHS kernel — x-axis anisotropy
 *
 * h1 gets: exchange (anisotropic: chb boosts x-neighbor coupling) + c_chk*m1
 * h2 gets: isotropic exchange only
 * h3 gets: isotropic exchange only
 *
 * LLG: dm/dt = chg*(m×h) + alpha*(h - (m·h)*m)
 */
__global__ static void f_kernel_group_soa_periodic(
    const sunrealtype* __restrict__ y,
    sunrealtype* __restrict__ yd,
    int ng, int ny, int ncell)
{
  const int gx = blockIdx.x*blockDim.x + threadIdx.x;
  const int gy = blockIdx.y*blockDim.y + threadIdx.y;
  if (gx>=ng || gy>=ny) return;

  const int cell = gy*ng+gx;
  const int mx = idx_mx(cell,ncell);
  const int my = idx_my(cell,ncell);
  const int mz = idx_mz(cell,ncell);

  const int xl   = wrap_x(gx-1,ng), xr   = wrap_x(gx+1,ng);
  const int yu   = wrap_y(gy-1,ny), ydwn = wrap_y(gy+1,ny);

  const int lc = gy*ng+xl, rc = gy*ng+xr;
  const int uc = yu*ng+gx, dc = ydwn*ng+gx;

  const sunrealtype m1 = y[mx], m2 = y[my], m3 = y[mz];

  const int lx=idx_mx(lc,ncell), rx=idx_mx(rc,ncell);
  const int ux=idx_mx(uc,ncell), dx=idx_mx(dc,ncell);
  const int ly=idx_my(lc,ncell), ry=idx_my(rc,ncell);
  const int uy=idx_my(uc,ncell), dy=idx_my(dc,ncell);
  const int lz=idx_mz(lc,ncell), rz=idx_mz(rc,ncell);
  const int uz=idx_mz(uc,ncell), dz=idx_mz(dc,ncell);

  /* h1: exchange + anisotropic chb (x-neighbors only) + x-anisotropy self */
  const sunrealtype h1 =
      c_che*(y[lx]+y[rx]+y[ux]+y[dx]) +
      c_msk[0]*(c_chk*m1+c_cha) +
      c_chb*c_nsk[0]*(y[lx]+y[rx]);

  /* h2: isotropic exchange only (c_msk[1]=0, c_nsk[1]=0) */
  const sunrealtype h2 =
      c_che*(y[ly]+y[ry]+y[uy]+y[dy]) +
      c_msk[1]*(c_chk*m1+c_cha) +
      c_chb*c_nsk[1]*(y[ly]+y[ry]);

  /* h3: isotropic exchange only (c_msk[2]=0, c_nsk[2]=0) */
  const sunrealtype h3 =
      c_che*(y[lz]+y[rz]+y[uz]+y[dz]) +
      c_msk[2]*(c_chk*m1+c_cha) +
      c_chb*c_nsk[2]*(y[lz]+y[rz]);

  const sunrealtype mh = m1*h1+m2*h2+m3*h3;

  yd[mx] = c_chg*(m3*h2-m2*h3) + c_alpha*(h1-mh*m1);
  yd[my] = c_chg*(m1*h3-m3*h1) + c_alpha*(h2-mh*m2);
  yd[mz] = c_chg*(m2*h1-m1*h2) + c_alpha*(h3-mh*m3);
}

__global__ static void demag_correction_kernel(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ hd,
    sunrealtype*       __restrict__ ydot,
    int ncell)
{
  const int c = blockIdx.x*blockDim.x+threadIdx.x;
  if (c>=ncell) return;
  const sunrealtype m1=y[c],m2=y[ncell+c],m3=y[2*ncell+c];
  const sunrealtype h1=hd[c],h2=hd[ncell+c],h3=hd[2*ncell+c];
  const sunrealtype mh=m1*h1+m2*h2+m3*h3;
  ydot[c]         += c_chg*(m3*h2-m2*h3)+c_alpha*(h1-mh*m1);
  ydot[ncell+c]   += c_chg*(m1*h3-m3*h1)+c_alpha*(h2-mh*m2);
  ydot[2*ncell+c] += c_chg*(m2*h1-m1*h2)+c_alpha*(h3-mh*m3);
}

static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
  (void)t;
  UserData* ud = (UserData*)user_data;
  sunrealtype* yd  = N_VGetDeviceArrayPointer_Cuda(y);
  sunrealtype* ydt = N_VGetDeviceArrayPointer_Cuda(ydot);

  dim3 block(BLOCK_X,BLOCK_Y);
  dim3 grid((ud->ng+block.x-1)/block.x,(ud->ny+block.y-1)/block.y);
  f_kernel_group_soa_periodic<<<grid,block>>>(yd,ydt,ud->ng,ud->ny,ud->ncell);

  if (ud->demag && DEMAG_STRENGTH>0.0) {
    cudaMemsetAsync(ud->d_hdmag,0,(size_t)3*ud->ncell*sizeof(sunrealtype),0);
    Demag_Apply(ud->demag,(const double*)yd,(double*)ud->d_hdmag);
    const int b=256,g=(ud->ncell+b-1)/b;
    demag_correction_kernel<<<g,b>>>(yd,ud->d_hdmag,ydt,ud->ncell);
  }
  if (cudaPeekAtLastError()!=cudaSuccess) {
    fprintf(stderr,"kernel launch error: %s\n",cudaGetErrorString(cudaPeekAtLastError()));
    return -1;
  }
  return 0;
}

static void PrintFinalStats(void* cvode_mem, SUNLinearSolver LS) {
  (void)LS;
  long int nst,nfe,nsetups,nni,ncfn,netf,nge,nli,nlcf,njv;
  CVodeGetNumSteps(cvode_mem,&nst);
  CVodeGetNumRhsEvals(cvode_mem,&nfe);
  CVodeGetNumLinSolvSetups(cvode_mem,&nsetups);
  CVodeGetNumErrTestFails(cvode_mem,&netf);
  CVodeGetNumNonlinSolvIters(cvode_mem,&nni);
  CVodeGetNumNonlinSolvConvFails(cvode_mem,&ncfn);
  CVodeGetNumGEvals(cvode_mem,&nge);
  CVodeGetNumLinIters(cvode_mem,&nli);
  CVodeGetNumLinConvFails(cvode_mem,&nlcf);
  CVodeGetNumJtimesEvals(cvode_mem,&njv);
  printf("\nFinal Statistics:\n");
  printf("nst=%-6ld nfe=%-6ld nsetups=%-6ld nni=%-6ld ncfn=%-6ld netf=%-6ld nge=%ld\n",
         nst,nfe,nsetups,nni,ncfn,netf,nge);
  printf("nli=%-6ld nlcf=%-6ld njv=%ld\n",nli,nlcf,njv);
}

#if ENABLE_OUTPUT
static void WriteFrame(FILE* fp, sunrealtype t,
                       int nx, int ny, int ng, int ncell, N_Vector y) {
  N_VCopyFromDevice_Cuda(y);
  sunrealtype* d = N_VGetHostArrayPointer_Cuda(y);
  fprintf(fp,"%f %d %d\n",(double)t,nx,ny);
  for (int jp=0;jp<ny;jp++)
    for (int ip=0;ip<ng;ip++) {
      int c=jp*ng+ip;
      fprintf(fp,"%f %f %f\n",
        (double)d[idx_mx(c,ncell)],
        (double)d[idx_my(c,ncell)],
        (double)d[idx_mz(c,ncell)]);
    }
  fprintf(fp,"\n");
}
static int ShouldWriteFrame(long int iout, sunrealtype t) {
  if (t<=SUN_RCONST(EARLY_SAVE_UNTIL)) return (iout%EARLY_SAVE_EVERY)==0;
  return (iout%LATE_SAVE_EVERY)==0;
}
#endif

int main(int argc, char* argv[]) {
  (void)argc;(void)argv;

  SUNContext sunctx=NULL;
  sunrealtype *ydata=NULL,*abstol_data=NULL;
  sunrealtype t=T0,tout=T1,ttotal=SUN_RCONST(T_TOTAL);
  N_Vector y=NULL,abstol=NULL;
  SUNLinearSolver LS=NULL;
  SUNNonlinearSolver NLS=NULL;
  void* cvode_mem=NULL;
  int retval;
  long int iout,NOUT;
  UserData udata;
  memset(&udata,0,sizeof(udata));
  int cell;
  cudaEvent_t start,stop;
  float elapsed=0.0f;

  const int nx=600,ny=128;
  if (nx%GROUPSIZE!=0){fprintf(stderr,"nx not multiple of GROUPSIZE\n");return 1;}
  const int ng=nx/GROUPSIZE, ncell=ng*ny, neq=3*ncell;

#if ENABLE_OUTPUT
  FILE* fp=fopen("output.txt","w");
  if(!fp){fprintf(stderr,"Error opening output file\n");return 1;}
  setvbuf(fp,NULL,_IOFBF,1<<20);
#else
  FILE* fp=NULL;(void)fp;
#endif

  udata.nx=nx;udata.ny=ny;udata.ng=ng;udata.ncell=ncell;udata.neq=neq;

  CHECK_SUNDIALS(SUNContext_Create(SUN_COMM_NULL,&sunctx));
  udata.pd=Precond_Create(ng,ny,ncell);
  if(!udata.pd){fprintf(stderr,"Precond_Create failed\n");return 1;}

  const double dstr=(double)DEMAG_STRENGTH,dthk=(double)DEMAG_THICK;
  if(dstr>0.0){
    udata.demag=Demag_Init(ng,ny,dthk,dstr);
    if(!udata.demag){fprintf(stderr,"Demag_Init failed\n");Precond_Destroy(udata.pd);return 1;}
    CHECK_CUDA(cudaMalloc((void**)&udata.d_hdmag,(size_t)3*ncell*sizeof(sunrealtype)));
  }

  y=N_VNew_Cuda(neq,sunctx);
  abstol=N_VNew_Cuda(neq,sunctx);
  if(!y||!abstol){fprintf(stderr,"N_VNew_Cuda failed\n");goto cleanup;}
  FusedNVec_Init(y);

  ydata=N_VGetHostArrayPointer_Cuda(y);
  abstol_data=N_VGetHostArrayPointer_Cuda(abstol);
  if(!ydata||!abstol_data){fprintf(stderr,"host pointer failed\n");goto cleanup;}

  /*
   * Initial condition — in-plane Neel-wall head-on transition
   *
   * phi(x) = pi * (1 - 0.5*(tanh((x-x1)/w) - tanh((x-x2)/w)))
   *
   * where x1=ng/4, x2=3*ng/4, w=WALL_WIDTH_FRAC*ng
   *
   * mx = cos(phi),  my = sin(phi),  mz = 0
   *
   * Verification:
   *   far left  (x<<x1):  tanh1→-1, tanh2→-1  → phi=pi*(1-0)=pi   → mx=-1, my=0  ✓
   *   far right (x>>x2):  tanh1→+1, tanh2→+1  → phi=pi*(1-0)=pi   → mx=-1, my=0  ✓
   *   middle   (x1<x<x2): tanh1→+1, tanh2→-1  → phi=pi*(1-1)=0    → mx=+1, my=0  ✓
   *   at x=x1:            tanh1→0,  tanh2→-1  → phi=pi*(1-0.5)=pi/2 → mx=0, my=1  (wall)
   *   at x=x2:            tanh1→+1, tanh2→0   → phi=pi*(1-0.5)=pi/2 → mx=0, my=1  (wall)
   *
   * The wall rotates through +y (Neel wall).
   * Professor's formula: my=sin(theta), mx=sqrt(1-my^2) = |cos(phi)|.
   * Here we use the signed cos(phi) to properly handle the sign of mx.
   */
  {
    const double x1 = 0.25*(double)ng;
    const double x2 = 0.75*(double)ng;
    const double w  = fmax(1.0, (double)WALL_WIDTH_FRAC*(double)ng);

    for (int j=0;j<ny;j++) {
      for (int i=0;i<ng;i++) {
        cell = j*ng+i;
        const int imx=idx_mx(cell,ncell);
        const int imy=idx_my(cell,ncell);
        const int imz=idx_mz(cell,ncell);

        const double xi = (double)i;
        const double t1 = tanh((xi-x1)/w);
        const double t2 = tanh((xi-x2)/w);

        /* phi: pi on outside, 0 in middle */
        const double phi = M_PI*(1.0 - 0.5*(t1-t2));

        const double mx0 = cos(phi);   /* -1 outside, +1 middle */
        const double my0 = sin(phi);   /* 0 outside, peaks ±1 at walls */
        /* mz = 0 exactly */

        ydata[imx] = SUN_RCONST(mx0);
        ydata[imy] = SUN_RCONST(my0);
        ydata[imz] = SUN_RCONST(0.0);

        abstol_data[imx]=ATOL1;
        abstol_data[imy]=ATOL2;
        abstol_data[imz]=ATOL3;
      }
    }
  }

  N_VCopyToDevice_Cuda(y);
  N_VCopyToDevice_Cuda(abstol);

#if ENABLE_OUTPUT
  WriteFrame(fp,T0,nx,ny,ng,ncell,y);
#endif

  cvode_mem=CVodeCreate(CV_BDF,sunctx);
  if(!cvode_mem){fprintf(stderr,"CVodeCreate failed\n");goto cleanup;}

  CHECK_SUNDIALS(CVodeInit(cvode_mem,f,T0,y));
  CHECK_SUNDIALS(CVodeSetUserData(cvode_mem,&udata));
  CHECK_SUNDIALS(CVodeSVtolerances(cvode_mem,RTOL,abstol));

  NLS=SUNNonlinSol_Newton(y,sunctx);
  if(!NLS){fprintf(stderr,"SUNNonlinSol_Newton failed\n");goto cleanup;}
  CHECK_SUNDIALS(CVodeSetNonlinearSolver(cvode_mem,NLS));

  LS=SUNLinSol_SPGMR(y,SUN_PREC_LEFT,KRYLOV_DIM,sunctx);
  if(!LS){fprintf(stderr,"SUNLinSol_SPGMR failed\n");goto cleanup;}
  CHECK_SUNDIALS(CVodeSetLinearSolver(cvode_mem,LS,NULL));
  CHECK_SUNDIALS(CVodeSetJacTimes(cvode_mem,NULL,JtvProduct));
  CHECK_SUNDIALS(CVodeSetPreconditioner(cvode_mem,PrecondSetup,PrecondSolve));

  if (neq<500000) {
    CHECK_SUNDIALS(SUNLinSol_SPGMRSetGSType(LS,SUN_CLASSICAL_GS));
    printf("GS type: Classical (neq=%d)\n",neq);
  } else {
    printf("GS type: Modified  (neq=%d)\n",neq);
  }
  CHECK_SUNDIALS(CVodeSetMaxOrd(cvode_mem,MAX_BDF_ORDER));

  printf("\n2D LLG head-on transition (Neel wall, x-axis anisotropy)\n");
  printf("nx=%d ny=%d ng=%d ncell=%d neq=%d\n",nx,ny,ng,ncell,neq);
  printf("init: phi=pi*(1-0.5*(tanh1-tanh2)), mx=cos(phi), my=sin(phi), mz=0\n");
  printf("      walls at x1=%.1f, x2=%.1f, width=%.2f cells\n",
         0.25*(double)ng, 0.75*(double)ng,
         fmax(1.0,(double)WALL_WIDTH_FRAC*(double)ng));
  printf("DEMAG_STRENGTH=%.4f DEMAG_THICK=%.4f\n",dstr,dthk);
  printf("T_TOTAL=%.2f RTOL/ATOL=%.1e\n",(double)T_TOTAL,(double)RTOL_VAL);

  NOUT=(long int)(ttotal/T1+SUN_RCONST(0.5));
  iout=0;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start,0));

  while (iout<NOUT) {
    retval=CVode(cvode_mem,tout,y,&t,CV_NORMAL);
    if(retval!=CV_SUCCESS){
      fprintf(stderr,"CVode error at output %ld: retval=%d\n",iout,retval);
      break;
    }
#if ENABLE_OUTPUT
    if(ShouldWriteFrame(iout+1,t))
      WriteFrame(fp,t,nx,ny,ng,ncell,y);
#endif
    iout++;tout+=T1;
  }

  CHECK_CUDA(cudaEventRecord(stop,0));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&elapsed,start,stop));
  printf("GPU simulation took %.3f ms\n",elapsed);
  PrintFinalStats(cvode_mem,LS);

cleanup:
  if(LS)        SUNLinSolFree(LS);
  if(NLS)       SUNNonlinSolFree(NLS);
  if(cvode_mem) CVodeFree(&cvode_mem);
  if(y)         N_VDestroy(y);
  if(abstol)    N_VDestroy(abstol);
  if(sunctx)    SUNContext_Free(&sunctx);
  Precond_Destroy(udata.pd);
  Demag_Destroy(udata.demag);
  if(udata.d_hdmag) cudaFree(udata.d_hdmag);
  FusedNVec_FreePool();
#if ENABLE_OUTPUT
  if(fp) fclose(fp);
#endif
  return 0;
}
