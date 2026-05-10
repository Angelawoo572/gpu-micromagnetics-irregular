/*
 * jtv.cu — Analytic Jv for 2D Maxwell on a collocated grid.
 *
 * Maxwell is LINEAR in (Ez, Hx, Hy), so J·v is just the same curl
 * operator applied to v instead of y.  Source term doesn't appear
 * (it's a constant in y).
 *
 *   (Jv)_Ez = c·( ∂vHy/∂x − ∂vHx/∂y )
 *   (Jv)_Hx = −c · ∂vEz/∂y
 *   (Jv)_Hy =  c · ∂vEz/∂x
 *
 * Same compact-launch + zero-inactive treatment as f().
 *
 * JtvUserData mirrors UserData in slit_fdtd.cu byte-for-byte.
 */

#include "jtv.h"
#include <cvode/cvode.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#ifndef C_LIGHT
#define C_LIGHT 2.99792458e8
#endif

#ifndef JTV_BLOCK_SIZE
#define JTV_BLOCK_SIZE 256
#endif

__device__ static inline int jidx_ez(int c, int nc) { return c; }
__device__ static inline int jidx_hx(int c, int nc) { return nc + c; }
__device__ static inline int jidx_hy(int c, int nc) { return 2*nc + c; }

__global__ static void jtv_zero_inactive_kernel(
    sunrealtype* __restrict__ Jv,
    const int* __restrict__ inactive_ids,
    int n_inactive,
    int ncell)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_inactive) return;
  int cell = inactive_ids[tid];
  Jv[jidx_ez(cell, ncell)] = SUN_RCONST(0.0);
  Jv[jidx_hx(cell, ncell)] = SUN_RCONST(0.0);
  Jv[jidx_hy(cell, ncell)] = SUN_RCONST(0.0);
}

__global__ static void jtv_kernel_compact(
    const sunrealtype* __restrict__ v,
    const int* __restrict__ active_ids,
    int n_active,
    sunrealtype* __restrict__ Jv,
    int nx, int ny, int ncell,
    double inv_dx, double c_speed)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_active) return;

  const int cell = active_ids[tid];
  const int gx = cell % nx;
  const int gy = cell / nx;

  const int iez = jidx_ez(cell, ncell);
  const int ihx = jidx_hx(cell, ncell);
  const int ihy = jidx_hy(cell, ncell);

  const sunrealtype ez_l = (gx > 0)    ? v[jidx_ez((gy)*nx + (gx-1), ncell)] : SUN_RCONST(0.0);
  const sunrealtype ez_r = (gx < nx-1) ? v[jidx_ez((gy)*nx + (gx+1), ncell)] : SUN_RCONST(0.0);
  const sunrealtype ez_u = (gy > 0)    ? v[jidx_ez((gy-1)*nx + (gx), ncell)] : SUN_RCONST(0.0);
  const sunrealtype ez_d = (gy < ny-1) ? v[jidx_ez((gy+1)*nx + (gx), ncell)] : SUN_RCONST(0.0);

  const sunrealtype hx_u = (gy > 0)    ? v[jidx_hx((gy-1)*nx + (gx), ncell)] : SUN_RCONST(0.0);
  const sunrealtype hx_d = (gy < ny-1) ? v[jidx_hx((gy+1)*nx + (gx), ncell)] : SUN_RCONST(0.0);

  const sunrealtype hy_l = (gx > 0)    ? v[jidx_hy((gy)*nx + (gx-1), ncell)] : SUN_RCONST(0.0);
  const sunrealtype hy_r = (gx < nx-1) ? v[jidx_hy((gy)*nx + (gx+1), ncell)] : SUN_RCONST(0.0);

  const sunrealtype half_inv_dx = SUN_RCONST(0.5) * (sunrealtype)inv_dx;
  const sunrealtype dHy_dx = (hy_r - hy_l) * half_inv_dx;
  const sunrealtype dHx_dy = (hx_d - hx_u) * half_inv_dx;
  const sunrealtype dEz_dx = (ez_r - ez_l) * half_inv_dx;
  const sunrealtype dEz_dy = (ez_d - ez_u) * half_inv_dx;

  Jv[iez] =  (sunrealtype)c_speed * (dHy_dx - dHx_dy);
  Jv[ihx] = -(sunrealtype)c_speed * dEz_dy;
  Jv[ihy] =  (sunrealtype)c_speed * dEz_dx;
}

typedef struct {
  void *pd_opaque;
  int  *d_active_ids;
  int  *d_inactive_ids;
  int nx, ny, ncell, neq;
  int n_active, n_inactive;
  int src_col, pad0;
  double inv_dx;
  double omega;
  double t_ramp;
} JtvUserData;

int JtvProduct(N_Vector v, N_Vector Jv, sunrealtype t,
               N_Vector y, N_Vector fy, void* user_data, N_Vector tmp)
{
  (void)t; (void)y; (void)fy; (void)tmp;
  const JtvUserData* ud = (const JtvUserData*)user_data;

  sunrealtype *Jvdata = N_VGetDeviceArrayPointer_Cuda(Jv);

  if (ud->n_inactive > 0) {
    int g0 = (ud->n_inactive + JTV_BLOCK_SIZE - 1) / JTV_BLOCK_SIZE;
    jtv_zero_inactive_kernel<<<g0, JTV_BLOCK_SIZE>>>(
        Jvdata, ud->d_inactive_ids, ud->n_inactive, ud->ncell);
  }

  if (ud->n_active > 0) {
    int g1 = (ud->n_active + JTV_BLOCK_SIZE - 1) / JTV_BLOCK_SIZE;
    jtv_kernel_compact<<<g1, JTV_BLOCK_SIZE>>>(
        N_VGetDeviceArrayPointer_Cuda(v),
        ud->d_active_ids, ud->n_active,
        Jvdata,
        ud->nx, ud->ny, ud->ncell,
        ud->inv_dx, C_LIGHT);
  }

  if (cudaPeekAtLastError() != cudaSuccess) {
    fprintf(stderr, "jtv failed: %s\n", cudaGetErrorString(cudaGetLastError()));
    return -1;
  }
  return 0;
}
