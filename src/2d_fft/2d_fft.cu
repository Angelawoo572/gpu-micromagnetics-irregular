/**
 * 2d_fft.cu  —  2D periodic LLG solver + professor's FFT demagnetization
 *
 * Demag field:
 *   h_dmag(i,j) = Σ_{m,n} N(i-m, j-n) · M(m,n)          [convolution]
 *               = IFFT[ N̂(k) · M̂(k) ]                    [via cuFFT Z2Z]
 *
 * N(r) computed by calt/ctt (professor's Newell analytic integrals,
 * 9×9 sub-cell averaging), FFT'd once at startup and stored on device.
 *
 * Enable:  make DEMAG_STRENGTH=1.0 DEMAG_THICK=1.0
 * Disable: make DEMAG_STRENGTH=0.0   (zero overhead in f())
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
#define T0    SUN_RCONST(0.0)
#define T1    SUN_RCONST(0.1)
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
#ifndef CIRCLE_CENTER_X_FRAC
#define CIRCLE_CENTER_X_FRAC 0.50
#endif
#ifndef CIRCLE_CENTER_Y_FRAC
#define CIRCLE_CENTER_Y_FRAC 0.50
#endif
#ifndef CIRCLE_RADIUS_FRAC_Y
#define CIRCLE_RADIUS_FRAC_Y 0.22
#endif
#ifndef TEXTURE_CORE_MZ
#define TEXTURE_CORE_MZ -0.998
#endif
#ifndef TEXTURE_OUTER_MZ
#define TEXTURE_OUTER_MZ 0.998
#endif
#ifndef TEXTURE_WIDTH_FRAC
#define TEXTURE_WIDTH_FRAC 0.35
#endif
#ifndef TEXTURE_EPS
#define TEXTURE_EPS 1.0e-12
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

__constant__ sunrealtype c_msk[3] = {0.0, 0.0, 1.0};
__constant__ sunrealtype c_nsk[3] = {1.0, 0.0, 0.0};
__constant__ sunrealtype c_chk   = 1.0;
__constant__ sunrealtype c_che   = 4.0;
__constant__ sunrealtype c_alpha = 0.2;
__constant__ sunrealtype c_chg   = 1.0;
__constant__ sunrealtype c_cha   = 0.0;
__constant__ sunrealtype c_chb   = 0.3;

#define CHECK_CUDA(call) do { \
    cudaError_t _e=(call); \
    if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(_e));exit(EXIT_FAILURE);} } while(0)

#define CHECK_SUNDIALS(call) do { \
    int _f=(call); \
    if(_f<0){fprintf(stderr,"SUNDIALS %s:%d: flag=%d\n",__FILE__,__LINE__,_f); \
    exit(EXIT_FAILURE);} } while(0)

/*
 * UserData — field ORDER IS CRITICAL for JtvUserData and PrecondData casts.
 *
 *  offset  0: pd        (PrecondData*)
 *  offset  8: demag     (DemagData*)
 *  offset 16: d_hdmag   (sunrealtype*)
 *  offset 24: nx,ny,ng,ncell,neq (ints)
 */
typedef struct {
    PrecondData  *pd;
    DemagData    *demag;
    sunrealtype  *d_hdmag;
    int nx, ny, ng, ncell, neq;
} UserData;

__host__ __device__ static inline int idx_mx(int c,int nc){return c;}
__host__ __device__ static inline int idx_my(int c,int nc){return nc+c;}
__host__ __device__ static inline int idx_mz(int c,int nc){return 2*nc+c;}
__host__ __device__ static inline int wrap_x(int x,int ng){
    return(x<0)?(x+ng):((x>=ng)?(x-ng):x);}
__host__ __device__ static inline int wrap_y(int y,int ny){
    return(y<0)?(y+ny):((y>=ny)?(y-ny):y);}

/* ── exchange RHS kernel ─────────────────────────────────────────────── */
__global__ static void f_kernel_group_soa_periodic(
    const sunrealtype* __restrict__ y,
    sunrealtype* __restrict__ yd,
    int ng, int ny, int ncell)
{
    const int gx=blockIdx.x*blockDim.x+threadIdx.x;
    const int gy=blockIdx.y*blockDim.y+threadIdx.y;
    if(gx>=ng||gy>=ny) return;
    const int cell=gy*ng+gx;
    const int mx=idx_mx(cell,ncell),my=idx_my(cell,ncell),mz=idx_mz(cell,ncell);
    const int xl=wrap_x(gx-1,ng),xr=wrap_x(gx+1,ng);
    const int yu=wrap_y(gy-1,ny),ydwn=wrap_y(gy+1,ny);
    const int lc=gy*ng+xl,rc=gy*ng+xr,uc=yu*ng+gx,dc=ydwn*ng+gx;
    const sunrealtype m1=y[mx],m2=y[my],m3=y[mz];
    const int lx=idx_mx(lc,ncell),rx=idx_mx(rc,ncell),ux=idx_mx(uc,ncell),dx=idx_mx(dc,ncell);
    const int ly=idx_my(lc,ncell),ry=idx_my(rc,ncell),uy=idx_my(uc,ncell),dy=idx_my(dc,ncell);
    const int lz=idx_mz(lc,ncell),rz=idx_mz(rc,ncell),uz=idx_mz(uc,ncell),dz=idx_mz(dc,ncell);
    const sunrealtype h1=c_che*(y[lx]+y[rx]+y[ux]+y[dx])+c_msk[0]*(c_chk*m3+c_cha)+c_chb*c_nsk[0]*(y[lx]+y[rx]);
    const sunrealtype h2=c_che*(y[ly]+y[ry]+y[uy]+y[dy])+c_msk[1]*(c_chk*m3+c_cha)+c_chb*c_nsk[1]*(y[ly]+y[ry]);
    const sunrealtype h3=c_che*(y[lz]+y[rz]+y[uz]+y[dz])+c_msk[2]*(c_chk*m3+c_cha)+c_chb*c_nsk[2]*(y[lz]+y[rz]);
    const sunrealtype mh=m1*h1+m2*h2+m3*h3;
    yd[mx]=c_chg*(m3*h2-m2*h3)+c_alpha*(h1-mh*m1);
    yd[my]=c_chg*(m1*h3-m3*h1)+c_alpha*(h2-mh*m2);
    yd[mz]=c_chg*(m2*h1-m1*h2)+c_alpha*(h3-mh*m3);
}

/* ── demag correction kernel ─────────────────────────────────────────── */
__global__ static void demag_correction_kernel(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ hd,
    sunrealtype*       __restrict__ ydot,
    int ncell)
{
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=ncell) return;
    const sunrealtype m1=y[c],m2=y[ncell+c],m3=y[2*ncell+c];
    const sunrealtype h1=hd[c],h2=hd[ncell+c],h3=hd[2*ncell+c];
    const sunrealtype mh=m1*h1+m2*h2+m3*h3;
    ydot[c]        +=c_chg*(m3*h2-m2*h3)+c_alpha*(h1-mh*m1);
    ydot[ncell+c]  +=c_chg*(m1*h3-m3*h1)+c_alpha*(h2-mh*m2);
    ydot[2*ncell+c]+=c_chg*(m2*h1-m1*h2)+c_alpha*(h3-mh*m3);
}

/* ── RHS ─────────────────────────────────────────────────────────────── */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
    (void)t;
    UserData* ud=(UserData*)user_data;
    sunrealtype* yd=N_VGetDeviceArrayPointer_Cuda(y);
    sunrealtype* ydd=N_VGetDeviceArrayPointer_Cuda(ydot);

    dim3 block(BLOCK_X,BLOCK_Y);
    dim3 grid((ud->ng+block.x-1)/block.x,(ud->ny+block.y-1)/block.y);
    f_kernel_group_soa_periodic<<<grid,block>>>(yd,ydd,ud->ng,ud->ny,ud->ncell);

    if(ud->demag && DEMAG_STRENGTH>0.0){
        cudaMemsetAsync(ud->d_hdmag,0,(size_t)3*ud->ncell*sizeof(sunrealtype),0);
        Demag_Apply(ud->demag,(const double*)yd,(double*)ud->d_hdmag);
        const int b=256,g=(ud->ncell+b-1)/b;
        demag_correction_kernel<<<g,b>>>(yd,ud->d_hdmag,ydd,ud->ncell);
    }

    cudaError_t e=cudaPeekAtLastError();
    if(e!=cudaSuccess){
        fprintf(stderr,">>> ERROR in f: %s\n",cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

static void PrintFinalStats(void* cm, SUNLinearSolver LS)
{
    (void)LS;
    long int nst,nfe,nsetups,nni,ncfn,netf,nge,nli,nlcf,njv;
    CVodeGetNumSteps(cm,&nst); CVodeGetNumRhsEvals(cm,&nfe);
    CVodeGetNumLinSolvSetups(cm,&nsetups); CVodeGetNumErrTestFails(cm,&netf);
    CVodeGetNumNonlinSolvIters(cm,&nni); CVodeGetNumNonlinSolvConvFails(cm,&ncfn);
    CVodeGetNumGEvals(cm,&nge); CVodeGetNumLinIters(cm,&nli);
    CVodeGetNumLinConvFails(cm,&nlcf); CVodeGetNumJtimesEvals(cm,&njv);
    printf("\nFinal Statistics:\n");
    printf("nst=%-6ld nfe=%-6ld nsetups=%-6ld nni=%-6ld ncfn=%-6ld netf=%-6ld nge=%ld\n",
           nst,nfe,nsetups,nni,ncfn,netf,nge);
    printf("nli=%-6ld nlcf=%-6ld njvevals=%ld\n",nli,nlcf,njv);
}

#if ENABLE_OUTPUT
static void WriteFrame(FILE* fp,sunrealtype t,int nx,int ny,int ng,int ncell,N_Vector y){
    N_VCopyFromDevice_Cuda(y);
    sunrealtype* yd=N_VGetHostArrayPointer_Cuda(y);
    fprintf(fp,"%f %d %d\n",(double)t,nx,ny);
    for(int j=0;j<ny;j++) for(int i=0;i<ng;i++){
        int c=j*ng+i;
        fprintf(fp,"%f %f %f\n",
            (double)yd[idx_mx(c,ncell)],(double)yd[idx_my(c,ncell)],(double)yd[idx_mz(c,ncell)]);
    }
    fprintf(fp,"\n");
}
static int ShouldWriteFrame(long int iout,sunrealtype t){
    return(t<=SUN_RCONST(EARLY_SAVE_UNTIL))?(iout%EARLY_SAVE_EVERY)==0:(iout%LATE_SAVE_EVERY)==0;
}
#endif

int main(int argc, char* argv[])
{
    (void)argc;(void)argv;
    SUNContext sunctx=NULL;
    sunrealtype *ydata=NULL,*abstol_data=NULL;
    sunrealtype t=T0,tout=T1,ttotal=SUN_RCONST(T_TOTAL);
    N_Vector y=NULL,abstol=NULL;
    SUNLinearSolver LS=NULL;
    SUNNonlinearSolver NLS=NULL;
    void* cvode_mem=NULL;
    int retval; long int iout,NOUT;
    UserData udata; memset(&udata,0,sizeof(udata));
    cudaEvent_t start,stop; float elapsed=0.0f;

    const int nx=3000,ny=1280;
    if(nx%GROUPSIZE!=0){fprintf(stderr,"nx must be multiple of %d\n",GROUPSIZE);return 1;}
    const int ng=nx/GROUPSIZE,ncell=ng*ny,neq=3*ncell;

#if ENABLE_OUTPUT
    FILE* fp=fopen("output.txt","w");
    if(!fp){fprintf(stderr,"Cannot open output.txt\n");return 1;}
    setvbuf(fp,NULL,_IOFBF,1<<20);
#else
    FILE* fp=NULL; (void)fp;
#endif

    const double cx=CIRCLE_CENTER_X_FRAC*(double)(ng-1);
    const double cy=CIRCLE_CENTER_Y_FRAC*(double)(ny-1);
    const double radius=CIRCLE_RADIUS_FRAC_Y*(double)ny;

    udata.nx=nx;udata.ny=ny;udata.ng=ng;udata.ncell=ncell;udata.neq=neq;

    CHECK_SUNDIALS(SUNContext_Create(SUN_COMM_NULL,&sunctx));

    udata.pd=Precond_Create(ng,ny,ncell);
    if(!udata.pd){fprintf(stderr,"Precond_Create failed\n");return 1;}

    const double dstr=(double)DEMAG_STRENGTH;
    const double dthk=(double)DEMAG_THICK;
    if(dstr>0.0){
        udata.demag=Demag_Init(ng,ny,dthk,dstr);
        if(!udata.demag){fprintf(stderr,"Demag_Init failed\n");Precond_Destroy(udata.pd);return 1;}
        CHECK_CUDA(cudaMalloc((void**)&udata.d_hdmag,(size_t)3*ncell*sizeof(sunrealtype)));
    }

    y=N_VNew_Cuda(neq,sunctx); abstol=N_VNew_Cuda(neq,sunctx);
    if(!y||!abstol){fprintf(stderr,"N_VNew_Cuda failed\n");goto cleanup;}
    FusedNVec_Init(y);
    ydata=N_VGetHostArrayPointer_Cuda(y);
    abstol_data=N_VGetHostArrayPointer_Cuda(abstol);
    if(!ydata||!abstol_data){fprintf(stderr,"host ptr failed\n");goto cleanup;}

    {
        const double core_mz=(double)TEXTURE_CORE_MZ;
        const double outer_mz=(double)TEXTURE_OUTER_MZ;
        double width=(double)TEXTURE_WIDTH_FRAC*radius;
        if(width<1.0) width=1.0;
        for(int j=0;j<ny;j++) for(int i=0;i<ng;i++){
            int cell=j*ng+i;
            const int mx=idx_mx(cell,ncell),my=idx_my(cell,ncell),mz=idx_mz(cell,ncell);
            double ddx=(double)i-cx,ddy=(double)j-cy;
            double rho=sqrt(ddx*ddx+ddy*ddy);
            double u=(rho-radius)/width, s=0.5*(1.0-tanh(u));
            double mz0=outer_mz+(core_mz-outer_mz)*s;
            if(mz0>1.0)mz0=1.0; if(mz0<-1.0)mz0=-1.0;
            double mperp=sqrt(fmax(0.0,1.0-mz0*mz0));
            double mx0,my0;
            if(rho>(double)TEXTURE_EPS){mx0=mperp*(ddx/rho);my0=mperp*(ddy/rho);}
            else{mx0=0.0;my0=0.0;}
            ydata[mx]=SUN_RCONST(mx0); ydata[my]=SUN_RCONST(my0); ydata[mz]=SUN_RCONST(mz0);
            abstol_data[mx]=ATOL1; abstol_data[my]=ATOL2; abstol_data[mz]=ATOL3;
        }
    }
    N_VCopyToDevice_Cuda(y); N_VCopyToDevice_Cuda(abstol);

#if ENABLE_OUTPUT
    WriteFrame(fp,T0,nx,ny,ng,ncell,y);
#endif

    cvode_mem=CVodeCreate(CV_BDF,sunctx);
    if(!cvode_mem){fprintf(stderr,"CVodeCreate failed\n");goto cleanup;}
    CHECK_SUNDIALS(CVodeInit(cvode_mem,f,T0,y));
    CHECK_SUNDIALS(CVodeSetUserData(cvode_mem,&udata));
    CHECK_SUNDIALS(CVodeSVtolerances(cvode_mem,RTOL,abstol));

    NLS=SUNNonlinSol_Newton(y,sunctx);
    if(!NLS){fprintf(stderr,"NLS failed\n");goto cleanup;}
    CHECK_SUNDIALS(CVodeSetNonlinearSolver(cvode_mem,NLS));

    LS=SUNLinSol_SPGMR(y,SUN_PREC_LEFT,KRYLOV_DIM,sunctx);
    if(!LS){fprintf(stderr,"LS failed\n");goto cleanup;}
    CHECK_SUNDIALS(CVodeSetLinearSolver(cvode_mem,LS,NULL));
    CHECK_SUNDIALS(CVodeSetJacTimes(cvode_mem,NULL,JtvProduct));
    CHECK_SUNDIALS(CVodeSetPreconditioner(cvode_mem,PrecondSetup,PrecondSolve));

    if(neq<500000){CHECK_SUNDIALS(SUNLinSol_SPGMRSetGSType(LS,SUN_CLASSICAL_GS));
        printf("GS: Classical (neq=%d)\n",neq);}
    else printf("GS: Modified (neq=%d)\n",neq);
    CHECK_SUNDIALS(CVodeSetMaxOrd(cvode_mem,MAX_BDF_ORDER));

    printf("\n2D LLG + FFT Demag [Newell calt/ctt, Z2Z]\n");
    printf("nx=%d ny=%d ng=%d ncell=%d neq=%d\n",nx,ny,ng,ncell,neq);
    printf("DEMAG_STRENGTH=%.4f  DEMAG_THICK=%.4f  (%s)\n",
           dstr,dthk,dstr>0.0?"active":"disabled");
    printf("T_TOTAL=%.2f  RTOL/ATOL=%.1e\n",(double)T_TOTAL,(double)RTOL_VAL);

    NOUT=(long int)(ttotal/T1+SUN_RCONST(0.5)); iout=0;
    CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start,0));

    while(iout<NOUT){
        retval=CVode(cvode_mem,tout,y,&t,CV_NORMAL);
        if(retval!=CV_SUCCESS){
            fprintf(stderr,"CVode error at %ld: retval=%d\n",iout,retval);break;}
#if ENABLE_OUTPUT
        if(ShouldWriteFrame(iout+1,t)) WriteFrame(fp,t,nx,ny,ng,ncell,y);
#endif
        iout++; tout+=T1;
    }

    CHECK_CUDA(cudaEventRecord(stop,0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed,start,stop));
    printf("GPU simulation took %.3f ms\n",elapsed);
    PrintFinalStats(cvode_mem,LS);

cleanup:
    if(LS)SUNLinSolFree(LS);
    if(NLS)SUNNonlinSolFree(NLS);
    if(cvode_mem)CVodeFree(&cvode_mem);
    if(y)N_VDestroy(y); if(abstol)N_VDestroy(abstol);
    if(sunctx)SUNContext_Free(&sunctx);
    Precond_Destroy(udata.pd);
    Demag_Destroy(udata.demag);
    if(udata.d_hdmag)cudaFree(udata.d_hdmag);
    FusedNVec_FreePool();
#if ENABLE_OUTPUT
    if(fp)fclose(fp);
#endif
    return 0;
}
