/*
 * precond.cu — 3×3 Block-Diagonal Jacobi, x-axis anisotropy
 *
 * ALL constants MUST match 2d_fft.cu and jtv.cu, including c_chb=0.3.
 *
 * H1 = (c_che+c_chb)*(m1_L+m1_R) + c_che*(m1_U+m1_D) + c_chk*m1
 * H2 = c_che*(all m2 neighbors)
 * H3 = c_che*(all m3 neighbors)
 *
 * d1 = H1 + c_chk*m1,  d2 = H2,  d3 = H3
 */

#include "precond.h"
#include <cvode/cvode.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__constant__ sunrealtype pc_chk   = 1.0;
__constant__ sunrealtype pc_che   = 4.0;
__constant__ sunrealtype pc_alpha = 0.2;
__constant__ sunrealtype pc_chg   = 1.0;
__constant__ sunrealtype pc_cha   = 0.0;
__constant__ sunrealtype pc_chb   = 0.3;   /* MUST match 2d_fft.cu */

__device__ static inline int pidx_mx(int c,int nc){return c;}
__device__ static inline int pidx_my(int c,int nc){return nc+c;}
__device__ static inline int pidx_mz(int c,int nc){return 2*nc+c;}
__device__ static inline int pwrap_x(int x,int ng){return(x<0)?(x+ng):((x>=ng)?(x-ng):x);}
__device__ static inline int pwrap_y(int y,int ny){return(y<0)?(y+ny):((y>=ny)?(y-ny):y);}

__global__ static void build_J_kernel(
    const sunrealtype* __restrict__ y,
    sunrealtype*       __restrict__ d_J,
    int ng, int ny, int ncell)
{
  const int gx=blockIdx.x*blockDim.x+threadIdx.x;
  const int gy=blockIdx.y*blockDim.y+threadIdx.y;
  if(gx>=ng||gy>=ny) return;

  const int cell=gy*ng+gx;
  const sunrealtype m1=y[pidx_mx(cell,ncell)];
  const sunrealtype m2=y[pidx_my(cell,ncell)];
  const sunrealtype m3=y[pidx_mz(cell,ncell)];

  const int xl=pwrap_x(gx-1,ng),xr=pwrap_x(gx+1,ng);
  const int yu=pwrap_y(gy-1,ny),ydn=pwrap_y(gy+1,ny);
  const int lc=gy*ng+xl,rc=gy*ng+xr,uc=yu*ng+gx,dc=ydn*ng+gx;

  const sunrealtype che_chb = pc_che + pc_chb;  /* 4.3 */

  const sunrealtype h1 =
      che_chb*(y[pidx_mx(lc,ncell)]+y[pidx_mx(rc,ncell)]) +
      pc_che *(y[pidx_mx(uc,ncell)]+y[pidx_mx(dc,ncell)]) +
      pc_chk*m1 + pc_cha;

  const sunrealtype h2 =
      pc_che*(y[pidx_my(lc,ncell)]+y[pidx_my(rc,ncell)]+
              y[pidx_my(uc,ncell)]+y[pidx_my(dc,ncell)]);

  const sunrealtype h3 =
      pc_che*(y[pidx_mz(lc,ncell)]+y[pidx_mz(rc,ncell)]+
              y[pidx_mz(uc,ncell)]+y[pidx_mz(dc,ncell)]);

  const sunrealtype mh = m1*h1+m2*h2+m3*h3;

  const sunrealtype d1 = h1 + pc_chk*m1;
  const sunrealtype d2 = h2;
  const sunrealtype d3 = h3;

  const int b=cell*9;
  d_J[b+0] = pc_alpha*(-d1*m1-mh);
  d_J[b+1] = -pc_chg*h3-pc_alpha*d2*m1;
  d_J[b+2] =  pc_chg*h2-pc_alpha*d3*m1;
  d_J[b+3] =  pc_chg*h3-pc_alpha*d1*m2;
  d_J[b+4] = pc_alpha*(-d2*m2-mh);
  d_J[b+5] = -pc_chg*h1-pc_alpha*d3*m2;
  d_J[b+6] = -pc_chg*h2-pc_alpha*d1*m3;
  d_J[b+7] =  pc_chg*h1-pc_alpha*d2*m3;
  d_J[b+8] = pc_alpha*(-d3*m3-mh);
}

__global__ static void build_Pinv_kernel(
    const sunrealtype* __restrict__ d_J,
    sunrealtype gamma,
    sunrealtype* __restrict__ d_Pinv,
    int ncell)
{
  const int cell=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(cell>=ncell) return;
  const int b=cell*9;
  const sunrealtype P00=1.0-gamma*d_J[b+0],P01=-gamma*d_J[b+1],P02=-gamma*d_J[b+2];
  const sunrealtype P10=-gamma*d_J[b+3],P11=1.0-gamma*d_J[b+4],P12=-gamma*d_J[b+5];
  const sunrealtype P20=-gamma*d_J[b+6],P21=-gamma*d_J[b+7],P22=1.0-gamma*d_J[b+8];
  const sunrealtype det=P00*(P11*P22-P12*P21)-P01*(P10*P22-P12*P20)+P02*(P10*P21-P11*P20);
  const sunrealtype id=(det!=0.0)?(1.0/det):1.0;
  d_Pinv[b+0]= id*(P11*P22-P12*P21);
  d_Pinv[b+1]=-id*(P01*P22-P02*P21);
  d_Pinv[b+2]= id*(P01*P12-P02*P11);
  d_Pinv[b+3]=-id*(P10*P22-P12*P20);
  d_Pinv[b+4]= id*(P00*P22-P02*P20);
  d_Pinv[b+5]=-id*(P00*P12-P02*P10);
  d_Pinv[b+6]= id*(P10*P21-P11*P20);
  d_Pinv[b+7]=-id*(P00*P21-P01*P20);
  d_Pinv[b+8]= id*(P00*P11-P01*P10);
}

__global__ static void psolve_kernel(
    const sunrealtype* __restrict__ r,
    sunrealtype*       __restrict__ z,
    const sunrealtype* __restrict__ Pinv,
    int ncell)
{
  const int cell=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(cell>=ncell) return;
  const int i0=pidx_mx(cell,ncell),i1=pidx_my(cell,ncell),i2=pidx_mz(cell,ncell);
  const sunrealtype r0=r[i0],r1=r[i1],r2=r[i2];
  const int b=cell*9;
  z[i0]=Pinv[b+0]*r0+Pinv[b+1]*r1+Pinv[b+2]*r2;
  z[i1]=Pinv[b+3]*r0+Pinv[b+4]*r1+Pinv[b+5]*r2;
  z[i2]=Pinv[b+6]*r0+Pinv[b+7]*r1+Pinv[b+8]*r2;
}

PrecondData* Precond_Create(int ng,int ny,int ncell)
{
  PrecondData*pd=(PrecondData*)malloc(sizeof(PrecondData));
  if(!pd){fprintf(stderr,"precond: malloc failed\n");return NULL;}
  pd->ng=ng;pd->ny=ny;pd->ncell=ncell;pd->last_gamma=0.0;
  pd->d_J=NULL;pd->d_Pinv=NULL;
  const size_t sz=(size_t)ncell*9*sizeof(sunrealtype);
  if(cudaMalloc((void**)&pd->d_J,sz)!=cudaSuccess||
     cudaMalloc((void**)&pd->d_Pinv,sz)!=cudaSuccess){
    fprintf(stderr,"precond: cudaMalloc failed\n");
    Precond_Destroy(pd);return NULL;
  }
  printf("[Precond] x-axis, c_chb=0.3. Storage: %zu MB each.\n",sz/(1024*1024));
  return pd;
}

void Precond_Destroy(PrecondData*pd)
{
  if(!pd)return;
  if(pd->d_J)cudaFree(pd->d_J);
  if(pd->d_Pinv)cudaFree(pd->d_Pinv);
  free(pd);
}

int PrecondSetup(sunrealtype t,N_Vector y,N_Vector fy,
                 sunbooleantype jok,sunbooleantype*jcurPtr,
                 sunrealtype gamma,void*user_data)
{
  (void)t;(void)fy;
  PrecondData*pd=*(PrecondData**)user_data;
  if(!jok){
    const sunrealtype*yd=N_VGetDeviceArrayPointer_Cuda(y);
    const dim3 b2(16,8),g2((pd->ng+15)/16,(pd->ny+7)/8);
    build_J_kernel<<<g2,b2>>>(yd,pd->d_J,pd->ng,pd->ny,pd->ncell);
    if(cudaPeekAtLastError()!=cudaSuccess){
      fprintf(stderr,"precond: build_J failed\n");return -1;}
    *jcurPtr=SUNTRUE;
  } else *jcurPtr=SUNFALSE;
  const int b1=256,g1=(pd->ncell+255)/256;
  build_Pinv_kernel<<<g1,b1>>>(pd->d_J,gamma,pd->d_Pinv,pd->ncell);
  if(cudaPeekAtLastError()!=cudaSuccess){
    fprintf(stderr,"precond: build_Pinv failed\n");return -1;}
  pd->last_gamma=gamma;
  return 0;
}

int PrecondSolve(sunrealtype t,N_Vector y,N_Vector fy,
                 N_Vector r,N_Vector z,
                 sunrealtype gamma,sunrealtype delta,int lr,void*user_data)
{
  (void)t;(void)y;(void)fy;(void)gamma;(void)delta;(void)lr;
  PrecondData*pd=*(PrecondData**)user_data;
  const int b=256,g=(pd->ncell+255)/256;
  psolve_kernel<<<g,b>>>(N_VGetDeviceArrayPointer_Cuda(r),
                         N_VGetDeviceArrayPointer_Cuda(z),
                         pd->d_Pinv,pd->ncell);
  if(cudaPeekAtLastError()!=cudaSuccess){
    fprintf(stderr,"precond: psolve failed\n");return -1;}
  return 0;
}
