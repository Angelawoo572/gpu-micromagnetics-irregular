/*
 * jtv.cu — Analytic Jv for exchange + per-component Landau anisotropy,
 *          compact active-cell execution (i6 variant).
 *
 * Standard simplified LLG form:
 *   dm/dt = γ (m × h) + α ( h − (m·h) m )
 *
 * ─── Compact-launch handling ────────────────────────────────────────
 *   1. zero_inactive_kernel: Jv[hole] = 0
 *   2. compact jtv_kernel:   one thread per active cell, reads
 *                            active_ids[tid] → cell.
 *
 * No ymsk multiplication, no `if (active[…])` branches.  Hole-cell
 * neighbor reads return 0 because m is frozen at 0 there, so the
 * formula is correct for active cells without any guarding.
 *
 * Demag is NOT included in this analytic Jv — adding it would require
 * a second FFT pipeline per Jv call.  GMRES + the preconditioner
 * tolerate the inexactness without issue.
 *
 * JtvUserData mirrors UserData in 2d_i6.cu byte-for-byte.
 */

#include "jtv.h"
#include <cvode/cvode.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__constant__ sunrealtype jc_chk   = 1.0;
__constant__ sunrealtype jc_che   = 10.0;
__constant__ sunrealtype jc_alpha = 0.2;
__constant__ sunrealtype jc_chg   = 1.0;

__device__ static inline int jidx_mx(int c,int nc){return c;}
__device__ static inline int jidx_my(int c,int nc){return nc+c;}
__device__ static inline int jidx_mz(int c,int nc){return 2*nc+c;}
__device__ static inline int jwrap_x(int x,int ng){return(x<0)?(x+ng):((x>=ng)?(x-ng):x);}
__device__ static inline int jwrap_y(int y,int ny){return(y<0)?(y+ny):((y>=ny)?(y-ny):y);}

#ifndef JTV_BLOCK_SIZE
#define JTV_BLOCK_SIZE 256
#endif

/* zero hole-cell entries of Jv before the compact launch */
__global__ static void jtv_zero_inactive_kernel(
    sunrealtype* __restrict__ Jv,
    const int* __restrict__ inactive_ids,
    int n_inactive,
    int ncell)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_inactive) return;
  int cell = inactive_ids[tid];
  Jv[jidx_mx(cell, ncell)] = SUN_RCONST(0.0);
  Jv[jidx_my(cell, ncell)] = SUN_RCONST(0.0);
  Jv[jidx_mz(cell, ncell)] = SUN_RCONST(0.0);
}

/* compact analytic Jv kernel: one thread per active cell */
__global__ static void jtv_kernel_compact(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ v,
    const int* __restrict__ active_ids,
    int n_active,
    sunrealtype* __restrict__ Jv,
    int ng, int ny, int ncell)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_active) return;

  const int cell = active_ids[tid];
  const int gx = cell % ng;
  const int gy = cell / ng;

  const int mx = jidx_mx(cell,ncell);
  const int my = jidx_my(cell,ncell);
  const int mz = jidx_mz(cell,ncell);

  const int xl=jwrap_x(gx-1,ng),xr=jwrap_x(gx+1,ng);
  const int yu=jwrap_y(gy-1,ny),ydn=jwrap_y(gy+1,ny);
  const int lc=gy*ng+xl,rc=gy*ng+xr,uc=yu*ng+gx,dc=ydn*ng+gx;

  const sunrealtype m1=y[mx],m2=y[my],m3=y[mz];
  const sunrealtype v1=v[mx],v2=v[my],v3=v[mz];

  /* Neighbor reads — no branching.  Hole-cell entries are 0. */
  const sunrealtype y1L=y[jidx_mx(lc,ncell)],y1R=y[jidx_mx(rc,ncell)];
  const sunrealtype y1U=y[jidx_mx(uc,ncell)],y1D=y[jidx_mx(dc,ncell)];
  const sunrealtype y2L=y[jidx_my(lc,ncell)],y2R=y[jidx_my(rc,ncell)];
  const sunrealtype y2U=y[jidx_my(uc,ncell)],y2D=y[jidx_my(dc,ncell)];
  const sunrealtype y3L=y[jidx_mz(lc,ncell)],y3R=y[jidx_mz(rc,ncell)];
  const sunrealtype y3U=y[jidx_mz(uc,ncell)],y3D=y[jidx_mz(dc,ncell)];

  const sunrealtype v1L=v[jidx_mx(lc,ncell)],v1R=v[jidx_mx(rc,ncell)];
  const sunrealtype v1U=v[jidx_mx(uc,ncell)],v1D=v[jidx_mx(dc,ncell)];
  const sunrealtype v2L=v[jidx_my(lc,ncell)],v2R=v[jidx_my(rc,ncell)];
  const sunrealtype v2U=v[jidx_my(uc,ncell)],v2D=v[jidx_my(dc,ncell)];
  const sunrealtype v3L=v[jidx_mz(lc,ncell)],v3R=v[jidx_mz(rc,ncell)];
  const sunrealtype v3U=v[jidx_mz(uc,ncell)],v3D=v[jidx_mz(dc,ncell)];

  /* Effective field h: exchange + per-component Landau anisotropy. */
  const sunrealtype h1 = jc_che * (y1L + y1R + y1U + y1D)
                       + jc_chk * m1 * (m1 * m1 - SUN_RCONST(1.0));
  const sunrealtype h2 = jc_che * (y2L + y2R + y2U + y2D)
                       + jc_chk * m2 * (m2 * m2 - SUN_RCONST(1.0));
  const sunrealtype h3 = jc_che * (y3L + y3R + y3U + y3D)
                       + jc_chk * m3 * (m3 * m3 - SUN_RCONST(1.0));

  const sunrealtype mh = m1*h1 + m2*h2 + m3*h3;

  /* dh/dm diagonal contribution from Landau anisotropy: 3 m_α² − 1. */
  const sunrealtype k1 = jc_chk * (SUN_RCONST(3.0) * m1 * m1 - SUN_RCONST(1.0));
  const sunrealtype k2 = jc_chk * (SUN_RCONST(3.0) * m2 * m2 - SUN_RCONST(1.0));
  const sunrealtype k3 = jc_chk * (SUN_RCONST(3.0) * m3 * m3 - SUN_RCONST(1.0));

  const sunrealtype dh1 = jc_che * (v1L + v1R + v1U + v1D) + k1 * v1;
  const sunrealtype dh2 = jc_che * (v2L + v2R + v2U + v2D) + k2 * v2;
  const sunrealtype dh3 = jc_che * (v3L + v3R + v3U + v3D) + k3 * v3;

  const sunrealtype dmh = (v1*h1 + v2*h2 + v3*h3) + (m1*dh1 + m2*dh2 + m3*dh3);

  Jv[mx] = jc_chg*(v3*h2 + m3*dh2 - v2*h3 - m2*dh3)
         + jc_alpha*(dh1 - dmh*m1 - mh*v1);
  Jv[my] = jc_chg*(v1*h3 + m1*dh3 - v3*h1 - m3*dh1)
         + jc_alpha*(dh2 - dmh*m2 - mh*v2);
  Jv[mz] = jc_chg*(v2*h1 + m2*dh1 - v1*h2 - m1*dh2)
         + jc_alpha*(dh3 - dmh*m3 - mh*v3);
}

/*
 * JtvUserData: byte-compatible mirror of UserData in 2d_i6.cu.
 *   offset 0  : void* pd_opaque
 *   offset 8  : void* demag_opaque
 *   offset 16 : void* d_hdmag_opaque
 *   offset 24 : int*  d_active_ids
 *   offset 32 : int*  d_inactive_ids
 *   offset 40 : int   nx, ny, ng, ncell, neq    (5*4 = 20 B)
 *   offset 60 : int   n_active, n_inactive      (2*4 = 8 B)
 *   offset 72 : double nxx0, nyy0, nzz0
 */
typedef struct {
  void *pd_opaque, *demag_opaque, *d_hdmag_opaque;
  int  *d_active_ids;
  int  *d_inactive_ids;
  int nx, ny, ng, ncell, neq;
  int n_active, n_inactive;
  double nxx0, nyy0, nzz0;
} JtvUserData;

int JtvProduct(N_Vector v, N_Vector Jv, sunrealtype t,
               N_Vector y, N_Vector fy, void* user_data, N_Vector tmp)
{
  (void)t; (void)fy; (void)tmp;
  const JtvUserData* ud = (const JtvUserData*)user_data;

  sunrealtype *Jvdata = N_VGetDeviceArrayPointer_Cuda(Jv);

  /* zero hole entries of Jv */
  if (ud->n_inactive > 0) {
    int g0 = (ud->n_inactive + JTV_BLOCK_SIZE - 1) / JTV_BLOCK_SIZE;
    jtv_zero_inactive_kernel<<<g0, JTV_BLOCK_SIZE>>>(
        Jvdata, ud->d_inactive_ids, ud->n_inactive, ud->ncell);
  }

  /* compact Jv at active cells */
  if (ud->n_active > 0) {
    int g1 = (ud->n_active + JTV_BLOCK_SIZE - 1) / JTV_BLOCK_SIZE;
    jtv_kernel_compact<<<g1, JTV_BLOCK_SIZE>>>(
        N_VGetDeviceArrayPointer_Cuda(y),
        N_VGetDeviceArrayPointer_Cuda(v),
        ud->d_active_ids, ud->n_active,
        Jvdata,
        ud->ng, ud->ny, ud->ncell);
  }

  if (cudaPeekAtLastError() != cudaSuccess) {
    fprintf(stderr,"jtv failed: %s\n",cudaGetErrorString(cudaGetLastError()));
    return -1;
  }
  return 0;
}
