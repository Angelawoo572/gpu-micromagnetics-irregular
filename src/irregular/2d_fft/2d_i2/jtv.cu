/*
 * jtv.cu — Analytic Jv for exchange + x-axis anisotropy + DMI.
 *
 * Uses the standard simplified LLG form:
 *
 *   dm/dt = γ (m × h) + α ( h − (m·h) m )
 *
 * which assumes |m|=1.  This is enforced by normalize_m_kernel in
 * 2d_fft.cu which projects y onto the unit sphere at the top of every
 * f() call, before CVODE evaluates the RHS or Jv.
 *
 * Demag is NOT included in this analytic Jv — adding it would require
 * a second FFT pipeline per Jv call.  The inexactness is handled by
 * the preconditioned Krylov iteration without issue.
 *
 * JtvUserData is a 3-pointer mirror matching UserData's first 3 pointers:
 *   offset 0  : void* pd_opaque
 *   offset 8  : void* demag_opaque
 *   offset 16 : void* d_hdmag_opaque
 *   offset 24 : int nx, ...
 */

#include "jtv.h"
#include <cvode/cvode.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__constant__ sunrealtype jc_chk   = 4.0;
__constant__ sunrealtype jc_che   = 4.0;
__constant__ sunrealtype jc_alpha = 0.2;
__constant__ sunrealtype jc_chg   = 1.0;
__constant__ sunrealtype jc_cha   = 0.0;
__constant__ sunrealtype jc_chb   = 0.3;

__device__ static inline int jidx_mx(int c,int nc){return c;}
__device__ static inline int jidx_my(int c,int nc){return nc+c;}
__device__ static inline int jidx_mz(int c,int nc){return 2*nc+c;}
__device__ static inline int jwrap_x(int x,int ng){return(x<0)?(x+ng):((x>=ng)?(x-ng):x);}
__device__ static inline int jwrap_y(int y,int ny){return(y<0)?(y+ny):((y>=ny)?(y-ny):y);}

__global__ static void jtv_kernel(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ v,
    sunrealtype*       __restrict__ Jv,
    const unsigned char* __restrict__ active,
    const int* __restrict__ active_ids,
    int n_active,
    int ng, int ny, int ncell)
{
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  if(tid>=n_active) return;

  const int cell=active_ids[tid];
  const int gx=cell%ng;
  const int gy=cell/ng;
  const int mx=jidx_mx(cell,ncell),my=jidx_my(cell,ncell),mz=jidx_mz(cell,ncell);

  const int xl=jwrap_x(gx-1,ng),xr=jwrap_x(gx+1,ng);
  const int yu=jwrap_y(gy-1,ny),ydn=jwrap_y(gy+1,ny);
  const int lc=gy*ng+xl,rc=gy*ng+xr,uc=yu*ng+gx,dc=ydn*ng+gx;

  const sunrealtype m1=y[mx],m2=y[my],m3=y[mz];
  const sunrealtype v1=v[mx],v2=v[my],v3=v[mz];

  const sunrealtype v1L=active[lc]?v[jidx_mx(lc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype v1R=active[rc]?v[jidx_mx(rc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype v1U=active[uc]?v[jidx_mx(uc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype v1D=active[dc]?v[jidx_mx(dc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype v2L=active[lc]?v[jidx_my(lc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype v2R=active[rc]?v[jidx_my(rc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype v2U=active[uc]?v[jidx_my(uc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype v2D=active[dc]?v[jidx_my(dc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype v3L=active[lc]?v[jidx_mz(lc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype v3R=active[rc]?v[jidx_mz(rc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype v3U=active[uc]?v[jidx_mz(uc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype v3D=active[dc]?v[jidx_mz(dc,ncell)]:SUN_RCONST(0.0);

  const sunrealtype che_chb = jc_che + jc_chb;

  const sunrealtype y1L=active[lc]?y[jidx_mx(lc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype y1R=active[rc]?y[jidx_mx(rc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype y1U=active[uc]?y[jidx_mx(uc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype y1D=active[dc]?y[jidx_mx(dc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype y2L=active[lc]?y[jidx_my(lc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype y2R=active[rc]?y[jidx_my(rc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype y2U=active[uc]?y[jidx_my(uc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype y2D=active[dc]?y[jidx_my(dc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype y3L=active[lc]?y[jidx_mz(lc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype y3R=active[rc]?y[jidx_mz(rc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype y3U=active[uc]?y[jidx_mz(uc,ncell)]:SUN_RCONST(0.0);
  const sunrealtype y3D=active[dc]?y[jidx_mz(dc,ncell)]:SUN_RCONST(0.0);

  const sunrealtype h1 = che_chb*(y1L+y1R) + jc_che*(y1U+y1D) + jc_chk*m1 + jc_cha;
  const sunrealtype h2 = jc_che*(y2L+y2R+y2U+y2D);
  const sunrealtype h3 = jc_che*(y3L+y3R+y3U+y3D);

  const sunrealtype mh = m1*h1+m2*h2+m3*h3;

  const sunrealtype dh1 = che_chb*(v1L+v1R)+jc_che*(v1U+v1D)+jc_chk*v1;
  const sunrealtype dh2 = jc_che*(v2L+v2R+v2U+v2D);
  const sunrealtype dh3 = jc_che*(v3L+v3R+v3U+v3D);

  const sunrealtype dmh = (v1*h1+v2*h2+v3*h3)+(m1*dh1+m2*dh2+m3*dh3);

  Jv[mx] = jc_chg*(v3*h2+m3*dh2-v2*h3-m2*dh3)+jc_alpha*(dh1-dmh*m1-mh*v1);
  Jv[my] = jc_chg*(v1*h3+m1*dh3-v3*h1-m3*dh1)+jc_alpha*(dh2-dmh*m2-mh*v2);
  Jv[mz] = jc_chg*(v2*h1+m2*dh1-v1*h2-m1*dh2)+jc_alpha*(dh3-dmh*m3-mh*v3);
}

typedef struct {
  void *pd_opaque, *demag_opaque, *d_hdmag_opaque;
  int nx, ny, ng, ncell, neq;
  double nxx0, nyy0, nzz0;
  unsigned char* h_active;
  unsigned char* d_active;
  int* h_active_ids;
  int* d_active_ids;
  int  n_active;
  int* h_inactive_ids;
  int* d_inactive_ids;
  int  n_inactive;
} JtvUserData;

__global__ static void jtv_zero_inactive_kernel(sunrealtype* Jv, const int* ids, int n, int ncell) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  int c = ids[tid];
  Jv[jidx_mx(c,ncell)] = SUN_RCONST(0.0);
  Jv[jidx_my(c,ncell)] = SUN_RCONST(0.0);
  Jv[jidx_mz(c,ncell)] = SUN_RCONST(0.0);
}

int JtvProduct(N_Vector v, N_Vector Jv, sunrealtype t,
               N_Vector y, N_Vector fy, void* user_data, N_Vector tmp)
{
  (void)t;(void)fy;(void)tmp;
  const JtvUserData* ud = (const JtvUserData*)user_data;
  sunrealtype* Jvdata = N_VGetDeviceArrayPointer_Cuda(Jv);
  if (ud->n_inactive > 0) {
    const int b0 = 256, g0 = (ud->n_inactive + b0 - 1) / b0;
    jtv_zero_inactive_kernel<<<g0,b0>>>(Jvdata, ud->d_inactive_ids, ud->n_inactive, ud->ncell);
  }
  const int block=256, grid=(ud->n_active+block-1)/block;
  jtv_kernel<<<grid,block>>>(
      N_VGetDeviceArrayPointer_Cuda(y),
      N_VGetDeviceArrayPointer_Cuda(v),
      Jvdata,
      ud->d_active, ud->d_active_ids, ud->n_active,
      ud->ng, ud->ny, ud->ncell);
  if (cudaPeekAtLastError() != cudaSuccess) {
    fprintf(stderr,"jtv failed: %s\n",cudaGetErrorString(cudaGetLastError()));
    return -1;
  }
  return 0;
}
