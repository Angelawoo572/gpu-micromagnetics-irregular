/**
 * 2D periodic smooth-texture LLG solver
 * CVODE + CUDA, SoA layout
 *
 * Exact dynamic-activity scheduling version
 *
 * Geometry / topology:
 *   - full regular 2D grid
 *   - periodic in x
 *   - periodic in y
 *
 * Scheduling:
 *   REGION_MODE = 0:
 *     uniform full-grid baseline
 *
 *   REGION_MODE = 1:
 *     dynamic exact tile scheduling
 *       - classify tile activity from current magnetization and neighbor mismatch
 *       - active tiles  : exact shared-memory stencil kernel
 *       - calm tiles    : exact direct global-memory stencil kernel
 *
 * Correctness:
 *   - both active and calm tiles use the same exact RHS mathematics
 *   - dynamic activity only changes scheduling / kernel path, not the PDE update
 *   - therefore REGION_MODE=1 preserves the same RHS as REGION_MODE=0
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

#define GROUPSIZE 3

#ifndef NX
#define NX 600
#endif

#ifndef NY
#define NY 128
#endif

#ifndef RTOL_VAL
#define RTOL_VAL 1.0e-5
#endif

#ifndef ATOL_VAL
#define ATOL_VAL 1.0e-5
#endif

#define RTOL  SUN_RCONST(RTOL_VAL)
#define ATOL1 SUN_RCONST(ATOL_VAL)
#define ATOL2 SUN_RCONST(ATOL_VAL)
#define ATOL3 SUN_RCONST(ATOL_VAL)

#define T0    SUN_RCONST(0.0)
#define T1    SUN_RCONST(0.1)
#define ZERO  SUN_RCONST(0.0)

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

#ifndef TILE_X
#define TILE_X 16
#endif

#ifndef TILE_Y
#define TILE_Y 8
#endif

#ifndef REGION_MODE
#define REGION_MODE 0
#endif

#ifndef HALO_W
#define HALO_W 1
#endif

#ifndef REBUILD_EVERY
#define REBUILD_EVERY 20
#endif

#ifndef ACTIVITY_THRESHOLD
#define ACTIVITY_THRESHOLD 0.15
#endif

#ifndef ACTIVE_FALLBACK_TOPK
#define ACTIVE_FALLBACK_TOPK 1
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

__constant__ sunrealtype c_msk[3] = {
    SUN_RCONST(0.0), SUN_RCONST(0.0), SUN_RCONST(1.0)};
__constant__ sunrealtype c_nsk[3] = {
    SUN_RCONST(1.0), SUN_RCONST(0.0), SUN_RCONST(0.0)};

__constant__ sunrealtype c_chk   = SUN_RCONST(1.0);
__constant__ sunrealtype c_che   = SUN_RCONST(4.0);
__constant__ sunrealtype c_alpha = SUN_RCONST(0.2);
__constant__ sunrealtype c_chg   = SUN_RCONST(1.0);
__constant__ sunrealtype c_cha   = SUN_RCONST(0.0);
__constant__ sunrealtype c_chb   = SUN_RCONST(0.3);

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t _err = (call);                                               \
    if (_err != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(_err));                                     \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

#define CHECK_SUNDIALS(call)                                                 \
  do {                                                                       \
    int _flag = (call);                                                      \
    if (_flag < 0) {                                                         \
      fprintf(stderr, "SUNDIALS error at %s:%d: flag = %d\n",                \
              __FILE__, __LINE__, _flag);                                    \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

typedef struct {
  int nx;
  int ny;
  int ng;
  int ncell;
  int neq;

  int ntx;
  int nty;
  int ntile;

  int rhs_calls;
  int n_active_tiles;
  int n_calm_tiles;

  int* d_active_tile_ix;
  int* d_active_tile_iy;
  int* d_calm_tile_ix;
  int* d_calm_tile_iy;

  sunrealtype* d_tile_score;
  sunrealtype* h_tile_score;

  int* h_active_tile_ix;
  int* h_active_tile_iy;
  int* h_calm_tile_ix;
  int* h_calm_tile_iy;
} UserData;

__host__ __device__ static inline int idx_mx(int cell, int ncell) { return cell; }
__host__ __device__ static inline int idx_my(int cell, int ncell) { return ncell + cell; }
__host__ __device__ static inline int idx_mz(int cell, int ncell) { return 2 * ncell + cell; }

__host__ __device__ static inline int wrap_x(int x, int ng) {
  return (x < 0) ? (x + ng) : ((x >= ng) ? (x - ng) : x);
}

__host__ __device__ static inline int wrap_y(int y, int ny) {
  return (y < 0) ? (y + ny) : ((y >= ny) ? (y - ny) : y);
}

__device__ static inline void compute_full_h_direct(
    const sunrealtype* __restrict__ y,
    int gx, int gy, int ng, int ny, int ncell,
    sunrealtype m1, sunrealtype m2, sunrealtype m3,
    sunrealtype* h1, sunrealtype* h2, sunrealtype* h3) {

  const int xl   = wrap_x(gx - 1, ng);
  const int xr   = wrap_x(gx + 1, ng);
  const int yu   = wrap_y(gy - 1, ny);
  const int ydwn = wrap_y(gy + 1, ny);

  const int left_cell  = gy * ng + xl;
  const int right_cell = gy * ng + xr;
  const int up_cell    = yu * ng + gx;
  const int down_cell  = ydwn * ng + gx;

  const int lx = idx_mx(left_cell,  ncell);
  const int rx = idx_mx(right_cell, ncell);
  const int ux = idx_mx(up_cell,    ncell);
  const int dx = idx_mx(down_cell,  ncell);

  const int ly = idx_my(left_cell,  ncell);
  const int ry = idx_my(right_cell, ncell);
  const int uy = idx_my(up_cell,    ncell);
  const int dy = idx_my(down_cell,  ncell);

  const int lz = idx_mz(left_cell,  ncell);
  const int rz = idx_mz(right_cell, ncell);
  const int uz = idx_mz(up_cell,    ncell);
  const int dz = idx_mz(down_cell,  ncell);

  *h1 = c_che * (y[lx] + y[rx] + y[ux] + y[dx]) +
        c_msk[0] * (c_chk * m3 + c_cha) +
        c_chb * c_nsk[0] * (y[lx] + y[rx]);

  *h2 = c_che * (y[ly] + y[ry] + y[uy] + y[dy]) +
        c_msk[1] * (c_chk * m3 + c_cha) +
        c_chb * c_nsk[1] * (y[ly] + y[ry]);

  *h3 = c_che * (y[lz] + y[rz] + y[uz] + y[dz]) +
        c_msk[2] * (c_chk * m3 + c_cha) +
        c_chb * c_nsk[2] * (y[lz] + y[rz]);
}

__global__ static void full_kernel_uniform(
    const sunrealtype* __restrict__ y,
    sunrealtype* __restrict__ yd,
    int ng, int ny, int ncell) {

  const int gx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gy = blockIdx.y * blockDim.y + threadIdx.y;
  if (gx >= ng || gy >= ny) return;

  const int cell = gy * ng + gx;
  const int mx = idx_mx(cell, ncell);
  const int my = idx_my(cell, ncell);
  const int mz = idx_mz(cell, ncell);

  const sunrealtype m1 = y[mx];
  const sunrealtype m2 = y[my];
  const sunrealtype m3 = y[mz];

  sunrealtype h1, h2, h3;
  compute_full_h_direct(y, gx, gy, ng, ny, ncell, m1, m2, m3, &h1, &h2, &h3);

  const sunrealtype mh = m1 * h1 + m2 * h2 + m3 * h3;
  yd[mx] = c_chg * (m3 * h2 - m2 * h3) + c_alpha * (h1 - mh * m1);
  yd[my] = c_chg * (m1 * h3 - m3 * h1) + c_alpha * (h2 - mh * m2);
  yd[mz] = c_chg * (m2 * h1 - m1 * h2) + c_alpha * (h3 - mh * m3);
}

__global__ static void classify_tile_activity_kernel(
    const sunrealtype* __restrict__ y,
    sunrealtype* __restrict__ tile_score,
    int ng, int ny, int ncell, int ntx, int nty) {

  const int txi = blockIdx.x;
  const int tyi = blockIdx.y;
  if (txi >= ntx || tyi >= nty) return;

  const int lx = threadIdx.x;
  const int ly = threadIdx.y;
  const int gx = txi * TILE_X + lx;
  const int gy = tyi * TILE_Y + ly;

  __shared__ sunrealtype s_metric[TILE_X * TILE_Y];

  sunrealtype metric = SUN_RCONST(0.0);
  if (gx < ng && gy < ny) {
    const int cell = gy * ng + gx;
    const int mx = idx_mx(cell, ncell);
    const int my = idx_my(cell, ncell);
    const int mz = idx_mz(cell, ncell);

    const sunrealtype m1 = y[mx];
    const sunrealtype m2 = y[my];
    const sunrealtype m3 = y[mz];

    const int xl = wrap_x(gx - 1, ng);
    const int xr = wrap_x(gx + 1, ng);
    const int yu = wrap_y(gy - 1, ny);
    const int ydwn = wrap_y(gy + 1, ny);

    const int cl = gy * ng + xl;
    const int cr = gy * ng + xr;
    const int cu = yu * ng + gx;
    const int cd = ydwn * ng + gx;

    const sunrealtype dlx = m1 - y[idx_mx(cl, ncell)];
    const sunrealtype dly = m2 - y[idx_my(cl, ncell)];
    const sunrealtype dlz = m3 - y[idx_mz(cl, ncell)];

    const sunrealtype drx = m1 - y[idx_mx(cr, ncell)];
    const sunrealtype dry = m2 - y[idx_my(cr, ncell)];
    const sunrealtype drz = m3 - y[idx_mz(cr, ncell)];

    const sunrealtype dux = m1 - y[idx_mx(cu, ncell)];
    const sunrealtype duy = m2 - y[idx_my(cu, ncell)];
    const sunrealtype duz = m3 - y[idx_mz(cu, ncell)];

    const sunrealtype ddx = m1 - y[idx_mx(cd, ncell)];
    const sunrealtype ddy = m2 - y[idx_my(cd, ncell)];
    const sunrealtype ddz = m3 - y[idx_mz(cd, ncell)];

    metric =
        dlx * dlx + dly * dly + dlz * dlz +
        drx * drx + dry * dry + drz * drz +
        dux * dux + duy * duy + duz * duz +
        ddx * ddx + ddy * ddy + ddz * ddz;
  }

  const int lid = ly * TILE_X + lx;
  s_metric[lid] = metric;
  __syncthreads();

  for (int stride = (TILE_X * TILE_Y) / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
      if (s_metric[lid + stride] > s_metric[lid]) {
        s_metric[lid] = s_metric[lid + stride];
      }
    }
    __syncthreads();
  }

  if (lid == 0) {
    const int tile_id = tyi * ntx + txi;
    tile_score[tile_id] = s_metric[0];
  }
}

__global__ static void exact_tiles_direct_kernel(
    const sunrealtype* __restrict__ y,
    sunrealtype* __restrict__ yd,
    int ng, int ny, int ncell,
    const int* __restrict__ tile_ix,
    const int* __restrict__ tile_iy,
    int ntiles) {

  const int tid = blockIdx.x;
  if (tid >= ntiles) return;

  const int base_x = tile_ix[tid] * TILE_X;
  const int base_y = tile_iy[tid] * TILE_Y;
  const int gx = base_x + threadIdx.x;
  const int gy = base_y + threadIdx.y;

  if (gx >= ng || gy >= ny) return;

  const int cell = gy * ng + gx;
  const int mx = idx_mx(cell, ncell);
  const int my = idx_my(cell, ncell);
  const int mz = idx_mz(cell, ncell);

  const sunrealtype m1 = y[mx];
  const sunrealtype m2 = y[my];
  const sunrealtype m3 = y[mz];

  sunrealtype h1, h2, h3;
  compute_full_h_direct(y, gx, gy, ng, ny, ncell, m1, m2, m3, &h1, &h2, &h3);

  const sunrealtype mh = m1 * h1 + m2 * h2 + m3 * h3;
  yd[mx] = c_chg * (m3 * h2 - m2 * h3) + c_alpha * (h1 - mh * m1);
  yd[my] = c_chg * (m1 * h3 - m3 * h1) + c_alpha * (h2 - mh * m2);
  yd[mz] = c_chg * (m2 * h1 - m1 * h2) + c_alpha * (h3 - mh * m3);
}

__global__ static void exact_active_tiles_shared_kernel(
    const sunrealtype* __restrict__ y,
    sunrealtype* __restrict__ yd,
    int ng, int ny, int ncell,
    const int* __restrict__ tile_ix,
    const int* __restrict__ tile_iy,
    int ntiles) {

  const int tid = blockIdx.x;
  if (tid >= ntiles) return;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int base_x = tile_ix[tid] * TILE_X;
  const int base_y = tile_iy[tid] * TILE_Y;
  const int gx = base_x + tx;
  const int gy = base_y + ty;

  constexpr int SX = TILE_X + 2 * HALO_W;
  constexpr int SY = TILE_Y + 2 * HALO_W;
  __shared__ sunrealtype smx[SY][SX];
  __shared__ sunrealtype smy[SY][SX];
  __shared__ sunrealtype smz[SY][SX];

  const int sx = tx + HALO_W;
  const int sy = ty + HALO_W;

  if (gx < ng && gy < ny) {
    const int cell = gy * ng + gx;
    smx[sy][sx] = y[idx_mx(cell, ncell)];
    smy[sy][sx] = y[idx_my(cell, ncell)];
    smz[sy][sx] = y[idx_mz(cell, ncell)];
  } else {
    smx[sy][sx] = SUN_RCONST(0.0);
    smy[sy][sx] = SUN_RCONST(0.0);
    smz[sy][sx] = SUN_RCONST(0.0);
  }

  if (tx < HALO_W) {
    const int gx_l = wrap_x(gx - HALO_W, ng);
    const int gx_r = wrap_x(base_x + TILE_X + tx, ng);
    const int gy_c = wrap_y(gy, ny);

    if (gy < ny) {
      const int cell_l = gy_c * ng + gx_l;
      const int cell_r = gy_c * ng + gx_r;
      smx[sy][sx - HALO_W] = y[idx_mx(cell_l, ncell)];
      smy[sy][sx - HALO_W] = y[idx_my(cell_l, ncell)];
      smz[sy][sx - HALO_W] = y[idx_mz(cell_l, ncell)];
      smx[sy][sx + TILE_X] = y[idx_mx(cell_r, ncell)];
      smy[sy][sx + TILE_X] = y[idx_my(cell_r, ncell)];
      smz[sy][sx + TILE_X] = y[idx_mz(cell_r, ncell)];
    }
  }

  if (ty < HALO_W) {
    const int gy_u = wrap_y(gy - HALO_W, ny);
    const int gy_d = wrap_y(base_y + TILE_Y + ty, ny);
    const int gx_c = wrap_x(gx, ng);

    if (gx < ng) {
      const int cell_u = gy_u * ng + gx_c;
      const int cell_d = gy_d * ng + gx_c;
      smx[sy - HALO_W][sx] = y[idx_mx(cell_u, ncell)];
      smy[sy - HALO_W][sx] = y[idx_my(cell_u, ncell)];
      smz[sy - HALO_W][sx] = y[idx_mz(cell_u, ncell)];
      smx[sy + TILE_Y][sx] = y[idx_mx(cell_d, ncell)];
      smy[sy + TILE_Y][sx] = y[idx_my(cell_d, ncell)];
      smz[sy + TILE_Y][sx] = y[idx_mz(cell_d, ncell)];
    }
  }

  if (tx < HALO_W && ty < HALO_W) {
    const int gx_l = wrap_x(gx - HALO_W, ng);
    const int gx_r = wrap_x(base_x + TILE_X + tx, ng);
    const int gy_u = wrap_y(gy - HALO_W, ny);
    const int gy_d = wrap_y(base_y + TILE_Y + ty, ny);

    const int c_ul = gy_u * ng + gx_l;
    const int c_ur = gy_u * ng + gx_r;
    const int c_dl = gy_d * ng + gx_l;
    const int c_dr = gy_d * ng + gx_r;

    smx[sy - HALO_W][sx - HALO_W] = y[idx_mx(c_ul, ncell)];
    smy[sy - HALO_W][sx - HALO_W] = y[idx_my(c_ul, ncell)];
    smz[sy - HALO_W][sx - HALO_W] = y[idx_mz(c_ul, ncell)];

    smx[sy - HALO_W][sx + TILE_X] = y[idx_mx(c_ur, ncell)];
    smy[sy - HALO_W][sx + TILE_X] = y[idx_my(c_ur, ncell)];
    smz[sy - HALO_W][sx + TILE_X] = y[idx_mz(c_ur, ncell)];

    smx[sy + TILE_Y][sx - HALO_W] = y[idx_mx(c_dl, ncell)];
    smy[sy + TILE_Y][sx - HALO_W] = y[idx_my(c_dl, ncell)];
    smz[sy + TILE_Y][sx - HALO_W] = y[idx_mz(c_dl, ncell)];

    smx[sy + TILE_Y][sx + TILE_X] = y[idx_mx(c_dr, ncell)];
    smy[sy + TILE_Y][sx + TILE_X] = y[idx_my(c_dr, ncell)];
    smz[sy + TILE_Y][sx + TILE_X] = y[idx_mz(c_dr, ncell)];
  }

  __syncthreads();

  if (gx >= ng || gy >= ny) return;

  const sunrealtype m1 = smx[sy][sx];
  const sunrealtype m2 = smy[sy][sx];
  const sunrealtype m3 = smz[sy][sx];

  const sunrealtype h1 =
      c_che * (smx[sy][sx - 1] + smx[sy][sx + 1] + smx[sy - 1][sx] + smx[sy + 1][sx]) +
      c_msk[0] * (c_chk * m3 + c_cha) +
      c_chb * c_nsk[0] * (smx[sy][sx - 1] + smx[sy][sx + 1]);

  const sunrealtype h2 =
      c_che * (smy[sy][sx - 1] + smy[sy][sx + 1] + smy[sy - 1][sx] + smy[sy + 1][sx]) +
      c_msk[1] * (c_chk * m3 + c_cha) +
      c_chb * c_nsk[1] * (smy[sy][sx - 1] + smy[sy][sx + 1]);

  const sunrealtype h3 =
      c_che * (smz[sy][sx - 1] + smz[sy][sx + 1] + smz[sy - 1][sx] + smz[sy + 1][sx]) +
      c_msk[2] * (c_chk * m3 + c_cha) +
      c_chb * c_nsk[2] * (smz[sy][sx - 1] + smz[sy][sx + 1]);

  const int cell = gy * ng + gx;
  const int mx = idx_mx(cell, ncell);
  const int my = idx_my(cell, ncell);
  const int mz = idx_mz(cell, ncell);

  const sunrealtype mh = m1 * h1 + m2 * h2 + m3 * h3;
  yd[mx] = c_chg * (m3 * h2 - m2 * h3) + c_alpha * (h1 - mh * m1);
  yd[my] = c_chg * (m1 * h3 - m3 * h1) + c_alpha * (h2 - mh * m2);
  yd[mz] = c_chg * (m2 * h1 - m1 * h2) + c_alpha * (h3 - mh * m3);
}

static int compare_score_desc(const void* a, const void* b) {
  const sunrealtype sa = ((const sunrealtype*)a)[0];
  const sunrealtype sb = ((const sunrealtype*)b)[0];
  if (sa < sb) return 1;
  if (sa > sb) return -1;
  return 0;
}

static void SetupDynamicTileScheduling(UserData* udata) {
  udata->ntx = (udata->ng + TILE_X - 1) / TILE_X;
  udata->nty = (udata->ny + TILE_Y - 1) / TILE_Y;
  udata->ntile = udata->ntx * udata->nty;
  udata->rhs_calls = 0;
  udata->n_active_tiles = 0;
  udata->n_calm_tiles = 0;

  CHECK_CUDA(cudaMalloc((void**)&udata->d_active_tile_ix, sizeof(int) * udata->ntile));
  CHECK_CUDA(cudaMalloc((void**)&udata->d_active_tile_iy, sizeof(int) * udata->ntile));
  CHECK_CUDA(cudaMalloc((void**)&udata->d_calm_tile_ix, sizeof(int) * udata->ntile));
  CHECK_CUDA(cudaMalloc((void**)&udata->d_calm_tile_iy, sizeof(int) * udata->ntile));
  CHECK_CUDA(cudaMalloc((void**)&udata->d_tile_score, sizeof(sunrealtype) * udata->ntile));

  udata->h_tile_score = (sunrealtype*)malloc(sizeof(sunrealtype) * udata->ntile);
  udata->h_active_tile_ix = (int*)malloc(sizeof(int) * udata->ntile);
  udata->h_active_tile_iy = (int*)malloc(sizeof(int) * udata->ntile);
  udata->h_calm_tile_ix = (int*)malloc(sizeof(int) * udata->ntile);
  udata->h_calm_tile_iy = (int*)malloc(sizeof(int) * udata->ntile);

  if (!udata->h_tile_score || !udata->h_active_tile_ix || !udata->h_active_tile_iy ||
      !udata->h_calm_tile_ix || !udata->h_calm_tile_iy) {
    fprintf(stderr, "Host dynamic scheduling allocation failed.\n");
    exit(EXIT_FAILURE);
  }
}

static void FreeDynamicTileScheduling(UserData* udata) {
  if (udata->d_active_tile_ix) CHECK_CUDA(cudaFree(udata->d_active_tile_ix));
  if (udata->d_active_tile_iy) CHECK_CUDA(cudaFree(udata->d_active_tile_iy));
  if (udata->d_calm_tile_ix) CHECK_CUDA(cudaFree(udata->d_calm_tile_ix));
  if (udata->d_calm_tile_iy) CHECK_CUDA(cudaFree(udata->d_calm_tile_iy));
  if (udata->d_tile_score) CHECK_CUDA(cudaFree(udata->d_tile_score));

  free(udata->h_tile_score);
  free(udata->h_active_tile_ix);
  free(udata->h_active_tile_iy);
  free(udata->h_calm_tile_ix);
  free(udata->h_calm_tile_iy);

  memset(&udata->d_active_tile_ix, 0, sizeof(int*));
  memset(&udata->d_active_tile_iy, 0, sizeof(int*));
  memset(&udata->d_calm_tile_ix, 0, sizeof(int*));
  memset(&udata->d_calm_tile_iy, 0, sizeof(int*));
  memset(&udata->d_tile_score, 0, sizeof(sunrealtype*));
  udata->h_tile_score = NULL;
  udata->h_active_tile_ix = NULL;
  udata->h_active_tile_iy = NULL;
  udata->h_calm_tile_ix = NULL;
  udata->h_calm_tile_iy = NULL;
}

static void RebuildTileListsDynamic(const sunrealtype* ydata, UserData* udata) {
  dim3 block(TILE_X, TILE_Y);
  dim3 grid((unsigned int)udata->ntx, (unsigned int)udata->nty, 1);
  classify_tile_activity_kernel<<<grid, block>>>(
      ydata, udata->d_tile_score, udata->ng, udata->ny, udata->ncell, udata->ntx, udata->nty);
  CHECK_CUDA(cudaPeekAtLastError());
  CHECK_CUDA(cudaMemcpy(udata->h_tile_score, udata->d_tile_score,
                        sizeof(sunrealtype) * udata->ntile, cudaMemcpyDeviceToHost));

  int na = 0;
  int nc = 0;
  int best_tile = 0;
  sunrealtype best_score = udata->h_tile_score[0];

  for (int tid = 0; tid < udata->ntile; tid++) {
    const int tx = tid % udata->ntx;
    const int ty = tid / udata->ntx;
    const sunrealtype s = udata->h_tile_score[tid];
    if (s > best_score) {
      best_score = s;
      best_tile = tid;
    }
    if (s >= SUN_RCONST(ACTIVITY_THRESHOLD)) {
      udata->h_active_tile_ix[na] = tx;
      udata->h_active_tile_iy[na] = ty;
      na++;
    } else {
      udata->h_calm_tile_ix[nc] = tx;
      udata->h_calm_tile_iy[nc] = ty;
      nc++;
    }
  }

  if (na == 0 && udata->ntile > 0) {
    const int tx = best_tile % udata->ntx;
    const int ty = best_tile / udata->ntx;
    udata->h_active_tile_ix[0] = tx;
    udata->h_active_tile_iy[0] = ty;
    na = 1;
    nc = 0;
    for (int tid = 0; tid < udata->ntile; tid++) {
      if (tid == best_tile) continue;
      udata->h_calm_tile_ix[nc] = tid % udata->ntx;
      udata->h_calm_tile_iy[nc] = tid / udata->ntx;
      nc++;
    }
  }

  udata->n_active_tiles = na;
  udata->n_calm_tiles = nc;

  if (na > 0) {
    CHECK_CUDA(cudaMemcpy(udata->d_active_tile_ix, udata->h_active_tile_ix,
                          sizeof(int) * na, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(udata->d_active_tile_iy, udata->h_active_tile_iy,
                          sizeof(int) * na, cudaMemcpyHostToDevice));
  }
  if (nc > 0) {
    CHECK_CUDA(cudaMemcpy(udata->d_calm_tile_ix, udata->h_calm_tile_ix,
                          sizeof(int) * nc, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(udata->d_calm_tile_iy, udata->h_calm_tile_iy,
                          sizeof(int) * nc, cudaMemcpyHostToDevice));
  }
}

static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
  (void)t;
  UserData* udata = (UserData*)user_data;
  sunrealtype* ydata    = N_VGetDeviceArrayPointer_Cuda(y);
  sunrealtype* ydotdata = N_VGetDeviceArrayPointer_Cuda(ydot);

  if (BLOCK_X * BLOCK_Y > 1024) {
    fprintf(stderr, "Invalid block size: BLOCK_X * BLOCK_Y = %d > 1024\n", BLOCK_X * BLOCK_Y);
    return -1;
  }
  if (TILE_X != BLOCK_X || TILE_Y != BLOCK_Y) {
    fprintf(stderr, "For this version require TILE_X==BLOCK_X and TILE_Y==BLOCK_Y.\n");
    return -1;
  }
#if HALO_W != 1
  fprintf(stderr, "This exact shared-memory version currently requires HALO_W == 1.\n");
  return -1;
#endif

#if REGION_MODE == 0
  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid((udata->ng + block.x - 1) / block.x,
            (udata->ny + block.y - 1) / block.y);
  full_kernel_uniform<<<grid, block>>>(ydata, ydotdata, udata->ng, udata->ny, udata->ncell);
#else
  if (udata->rhs_calls == 0 || (REBUILD_EVERY > 0 && (udata->rhs_calls % REBUILD_EVERY) == 0)) {
    RebuildTileListsDynamic(ydata, udata);
  }

  dim3 block(TILE_X, TILE_Y);
  if (udata->n_active_tiles > 0) {
    dim3 grid_active((unsigned int)udata->n_active_tiles, 1, 1);
    exact_active_tiles_shared_kernel<<<grid_active, block>>>(
        ydata, ydotdata, udata->ng, udata->ny, udata->ncell,
        udata->d_active_tile_ix, udata->d_active_tile_iy, udata->n_active_tiles);
  }
  if (udata->n_calm_tiles > 0) {
    dim3 grid_calm((unsigned int)udata->n_calm_tiles, 1, 1);
    exact_tiles_direct_kernel<<<grid_calm, block>>>(
        ydata, ydotdata, udata->ng, udata->ny, udata->ncell,
        udata->d_calm_tile_ix, udata->d_calm_tile_iy, udata->n_calm_tiles);
  }
  udata->rhs_calls++;
#endif

  cudaError_t cuerr = cudaPeekAtLastError();
  if (cuerr != cudaSuccess) {
    fprintf(stderr, ">>> ERROR in f: kernel launch failed: %s\n", cudaGetErrorString(cuerr));
    return -1;
  }
  return 0;
}

static void PrintFinalStats(void* cvode_mem, SUNLinearSolver LS) {
  (void)LS;
  long int nst, nfe, nsetups, nni, ncfn, netf, nge;
  CVodeGetNumSteps(cvode_mem, &nst);
  CVodeGetNumRhsEvals(cvode_mem, &nfe);
  CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
  CVodeGetNumErrTestFails(cvode_mem, &netf);
  CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
  CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
  CVodeGetNumGEvals(cvode_mem, &nge);
  printf("\nFinal Statistics:\n");
  printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld ", nst, nfe, nsetups);
  printf("nni = %-6ld ncfn = %-6ld netf = %-6ld nge = %ld\n", nni, ncfn, netf, nge);
}

#if ENABLE_OUTPUT
static void WriteFrame(FILE* fp,
                       sunrealtype t,
                       int nx, int ny, int ng, int ncell,
                       N_Vector y) {
  N_VCopyFromDevice_Cuda(y);
  sunrealtype* ydata = N_VGetHostArrayPointer_Cuda(y);
  fprintf(fp, "%f %d %d\n", (double)t, nx, ny);
  for (int jp = 0; jp < ny; jp++) {
    for (int ip = 0; ip < ng; ip++) {
      int cell_out = jp * ng + ip;
      fprintf(fp, "%f %f %f\n",
              (double)ydata[idx_mx(cell_out, ncell)],
              (double)ydata[idx_my(cell_out, ncell)],
              (double)ydata[idx_mz(cell_out, ncell)]);
    }
  }
  fprintf(fp, "\n");
}

static int ShouldWriteFrame(long int iout, sunrealtype t) {
  if (t <= SUN_RCONST(EARLY_SAVE_UNTIL)) return (iout % EARLY_SAVE_EVERY) == 0;
  return (iout % LATE_SAVE_EVERY) == 0;
}
#endif

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  SUNContext sunctx = NULL;
  sunrealtype *ydata = NULL, *abstol_data = NULL;
  sunrealtype t = T0, tout = T1;
  sunrealtype ttotal = SUN_RCONST(T_TOTAL);

  N_Vector y = NULL, abstol = NULL;
  SUNLinearSolver LS = NULL;
  SUNNonlinearSolver NLS = NULL;
  void* cvode_mem = NULL;

  int retval;
  long int iout, NOUT;
  UserData udata;
  memset(&udata, 0, sizeof(udata));

  int cell;
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;

  const int nx = NX;
  const int ny = NY;
  if (nx % GROUPSIZE != 0) {
    fprintf(stderr, "nx must be a multiple of GROUPSIZE=%d\n", GROUPSIZE);
    return 1;
  }

  const int ng    = nx / GROUPSIZE;
  const int ncell = ng * ny;
  const int neq   = 3 * ncell;

#if ENABLE_OUTPUT
  FILE* fp = fopen("output.txt", "w");
  if (fp == NULL) {
    fprintf(stderr, "Error opening output file.\n");
    return 1;
  }
  setvbuf(fp, NULL, _IOFBF, 1 << 20);
#endif

  const sunrealtype cx = SUN_RCONST(CIRCLE_CENTER_X_FRAC) * (sunrealtype)(ng - 1);
  const sunrealtype cy = SUN_RCONST(CIRCLE_CENTER_Y_FRAC) * (sunrealtype)(ny - 1);
  const sunrealtype radius = SUN_RCONST(CIRCLE_RADIUS_FRAC_Y) * (sunrealtype)ny;

  udata.nx    = nx;
  udata.ny    = ny;
  udata.ng    = ng;
  udata.ncell = ncell;
  udata.neq   = neq;

  CHECK_SUNDIALS(SUNContext_Create(SUN_COMM_NULL, &sunctx));

  y      = N_VNew_Cuda(neq, sunctx);
  abstol = N_VNew_Cuda(neq, sunctx);
  if (y == NULL || abstol == NULL) {
    fprintf(stderr, "Failed to allocate N_Vector_Cuda objects.\n");
#if ENABLE_OUTPUT
    fclose(fp);
#endif
    return 1;
  }

  ydata       = N_VGetHostArrayPointer_Cuda(y);
  abstol_data = N_VGetHostArrayPointer_Cuda(abstol);
  if (ydata == NULL || abstol_data == NULL) {
    fprintf(stderr, "Failed to get host array pointers from N_Vector_Cuda.\n");
#if ENABLE_OUTPUT
    fclose(fp);
#endif
    N_VDestroy(y);
    N_VDestroy(abstol);
    return 1;
  }

  {
    const double core_mz  = (double)TEXTURE_CORE_MZ;
    const double outer_mz = (double)TEXTURE_OUTER_MZ;
    double width = (double)TEXTURE_WIDTH_FRAC * (double)radius;
    if (width < 1.0) width = 1.0;

    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < ng; i++) {
        cell = j * ng + i;
        const int mx = idx_mx(cell, ncell);
        const int my = idx_my(cell, ncell);
        const int mz = idx_mz(cell, ncell);

        const double dx = (double)i - (double)cx;
        const double dy = (double)j - (double)cy;
        const double rho = sqrt(dx * dx + dy * dy);
        const double u = (rho - (double)radius) / width;
        const double s = 0.5 * (1.0 - tanh(u));

        double mz0 = outer_mz + (core_mz - outer_mz) * s;
        if (mz0 >  1.0) mz0 =  1.0;
        if (mz0 < -1.0) mz0 = -1.0;

        double mperp = sqrt(fmax(0.0, 1.0 - mz0 * mz0));
        double mx0, my0;
        if (rho > (double)TEXTURE_EPS) {
          mx0 = mperp * (dx / rho);
          my0 = mperp * (dy / rho);
        } else {
          mx0 = 0.0;
          my0 = 0.0;
        }

        ydata[mx] = SUN_RCONST(mx0);
        ydata[my] = SUN_RCONST(my0);
        ydata[mz] = SUN_RCONST(mz0);

        abstol_data[mx] = ATOL1;
        abstol_data[my] = ATOL2;
        abstol_data[mz] = ATOL3;
      }
    }
  }

  N_VCopyToDevice_Cuda(y);
  N_VCopyToDevice_Cuda(abstol);
  N_VEnableFusedOps_Cuda(y, SUNTRUE);
  N_VEnableFusedOps_Cuda(abstol, SUNTRUE);

#if REGION_MODE == 1
  SetupDynamicTileScheduling(&udata);
#endif

#if ENABLE_OUTPUT
  WriteFrame(fp, T0, nx, ny, ng, ncell, y);
#endif

  cvode_mem = CVodeCreate(CV_BDF, sunctx);
  if (cvode_mem == NULL) {
    fprintf(stderr, "CVodeCreate failed.\n");
    goto cleanup;
  }

  CHECK_SUNDIALS(CVodeInit(cvode_mem, f, T0, y));
  CHECK_SUNDIALS(CVodeSetUserData(cvode_mem, &udata));
  CHECK_SUNDIALS(CVodeSVtolerances(cvode_mem, RTOL, abstol));

  NLS = SUNNonlinSol_Newton(y, sunctx);
  if (NLS == NULL) {
    fprintf(stderr, "SUNNonlinSol_Newton failed.\n");
    goto cleanup;
  }
  CHECK_SUNDIALS(CVodeSetNonlinearSolver(cvode_mem, NLS));

  LS = SUNLinSol_SPGMR(y, SUN_PREC_NONE, 0, sunctx);
  if (LS == NULL) {
    fprintf(stderr, "SUNLinSol_SPGMR failed.\n");
    goto cleanup;
  }
  CHECK_SUNDIALS(CVodeSetLinearSolver(cvode_mem, LS, NULL));

  printf("\n2D periodic smooth-texture solver (SoA)\n\n");
  printf("scalar width nx = %d, rows ny = %d, groups/row = %d, ncell = %d, neq = %d\n",
         nx, ny, ng, ncell, neq);
  printf("periodic BC: x and y\n");
  printf("circle center = (%.2f, %.2f), radius = %.2f cells\n", (double)cx, (double)cy, (double)radius);
  printf("smooth radial texture init\n");
  printf("core mz         = %.4f\n", (double)TEXTURE_CORE_MZ);
  printf("outer mz        = %.4f\n", (double)TEXTURE_OUTER_MZ);
  printf("width frac      = %.4f\n", (double)TEXTURE_WIDTH_FRAC);
  printf("T_TOTAL         = %.2f\n", (double)T_TOTAL);
  printf("REGION_MODE     = %d\n", REGION_MODE);
#if REGION_MODE == 1
  printf("ACTIVITY_THRESHOLD = %.6f\n", (double)ACTIVITY_THRESHOLD);
  printf("REBUILD_EVERY      = %d\n", REBUILD_EVERY);
  printf("HALO_W             = %d\n", HALO_W);
  printf("tile size          = %d x %d\n", TILE_X, TILE_Y);
#endif
#if ENABLE_OUTPUT
  printf("EARLY_SAVE_UNTIL = %.2f\n", (double)EARLY_SAVE_UNTIL);
  printf("EARLY_SAVE_EVERY = %d\n", EARLY_SAVE_EVERY);
  printf("LATE_SAVE_EVERY  = %d\n", LATE_SAVE_EVERY);
#endif

  NOUT = (long int)(ttotal / T1 + SUN_RCONST(0.5));
  iout = 0;

  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start, 0));

  while (iout < NOUT) {
    retval = CVode(cvode_mem, tout, y, &t, CV_NORMAL);
    if (retval != CV_SUCCESS) {
      fprintf(stderr, "CVode error at output %ld: retval = %d\n", iout, retval);
      break;
    }

#if ENABLE_OUTPUT
    if (ShouldWriteFrame(iout + 1, t)) {
      WriteFrame(fp, t, nx, ny, ng, ncell, y);
    }
#endif

    iout++;
    tout += T1;
  }

  CHECK_CUDA(cudaEventRecord(stop, 0));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("GPU simulation took %.3f ms\n", elapsedTime);

  PrintFinalStats(cvode_mem, LS);

cleanup:
#if REGION_MODE == 1
  FreeDynamicTileScheduling(&udata);
#endif
  if (LS) SUNLinSolFree(LS);
  if (NLS) SUNNonlinSolFree(NLS);
  if (cvode_mem) CVodeFree(&cvode_mem);
  if (y) N_VDestroy(y);
  if (abstol) N_VDestroy(abstol);
  if (sunctx) SUNContext_Free(&sunctx);
#if ENABLE_OUTPUT
  if (fp) fclose(fp);
#endif
  return 0;
}
