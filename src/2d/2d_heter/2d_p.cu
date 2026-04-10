/**
 * 2D periodic smooth-texture LLG solver
 * CVODE + CUDA, SoA layout
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
 *     static ring-aware tile partition with halo-aware binary classification
 *       - full tiles  : full stencil update everywhere in tile
 *       - cheap tiles : cheap approximation everywhere in tile
 *
 * Important difference from the older smooth-blend version:
 *   - no per-cell full/cheap blend inside a tile
 *   - classification is tile-level and conservative w.r.t. neighbor interaction
 *   - a tile is promoted to FULL if its tile footprint expanded by HALO_W and
 *     TILE_SAFETY intersects the active ring band
 *
 * CHEAP_MODE:
 *   0 = self-stencil approximation
 *   1 = zero derivative in cheap tiles (aggressive)
 *
 * NOTE:
 *   The cheap path is a heuristic approximation for performance experiments.
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

#ifndef CIRCLE_CENTER_X_FRAC
#define CIRCLE_CENTER_X_FRAC 0.50
#endif

#ifndef CIRCLE_CENTER_Y_FRAC
#define CIRCLE_CENTER_Y_FRAC 0.50
#endif

#ifndef CIRCLE_RADIUS_FRAC_Y
#define CIRCLE_RADIUS_FRAC_Y 0.22
#endif

#ifndef ACTIVE_RING_RADIUS_FRAC_Y
#define ACTIVE_RING_RADIUS_FRAC_Y CIRCLE_RADIUS_FRAC_Y
#endif

#ifndef ACTIVE_RING_HALF_WIDTH_FRAC_Y
#define ACTIVE_RING_HALF_WIDTH_FRAC_Y 0.08
#endif

#ifndef CHEAP_MODE
#define CHEAP_MODE 0
#endif

#ifndef HALO_W
#define HALO_W 1
#endif

#ifndef TILE_SAFETY_FRAC_Y
#define TILE_SAFETY_FRAC_Y 0.02
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

  sunrealtype cx;
  sunrealtype cy;
  sunrealtype active_ring_radius;
  sunrealtype active_ring_half_width;
  sunrealtype tile_safety;

  int ntx;
  int nty;
  int ntile;
  int n_full_tiles;
  int n_cheap_tiles;

  int* d_full_tile_ix;
  int* d_full_tile_iy;
  int* d_cheap_tile_ix;
  int* d_cheap_tile_iy;
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

__device__ static inline void compute_full_h(
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

__device__ static inline void compute_cheap_h(
    sunrealtype m1, sunrealtype m2, sunrealtype m3,
    sunrealtype* h1, sunrealtype* h2, sunrealtype* h3) {
#if CHEAP_MODE == 1
  *h1 = SUN_RCONST(0.0);
  *h2 = SUN_RCONST(0.0);
  *h3 = SUN_RCONST(0.0);
#else
  *h1 = c_che * (m1 + m1 + m1 + m1) +
        c_msk[0] * (c_chk * m3 + c_cha) +
        c_chb * c_nsk[0] * (m1 + m1);

  *h2 = c_che * (m2 + m2 + m2 + m2) +
        c_msk[1] * (c_chk * m3 + c_cha) +
        c_chb * c_nsk[1] * (m2 + m2);

  *h3 = c_che * (m3 + m3 + m3 + m3) +
        c_msk[2] * (c_chk * m3 + c_cha) +
        c_chb * c_nsk[2] * (m3 + m3);
#endif
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
  compute_full_h(y, gx, gy, ng, ny, ncell, m1, m2, m3, &h1, &h2, &h3);

  const sunrealtype mh = m1 * h1 + m2 * h2 + m3 * h3;
  yd[mx] = c_chg * (m3 * h2 - m2 * h3) + c_alpha * (h1 - mh * m1);
  yd[my] = c_chg * (m1 * h3 - m3 * h1) + c_alpha * (h2 - mh * m2);
  yd[mz] = c_chg * (m2 * h1 - m1 * h2) + c_alpha * (h3 - mh * m3);
}

__global__ static void full_tiles_kernel(
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

  if (threadIdx.x >= TILE_X || threadIdx.y >= TILE_Y) return;
  if (gx >= ng || gy >= ny) return;

  const int cell = gy * ng + gx;
  const int mx = idx_mx(cell, ncell);
  const int my = idx_my(cell, ncell);
  const int mz = idx_mz(cell, ncell);

  const sunrealtype m1 = y[mx];
  const sunrealtype m2 = y[my];
  const sunrealtype m3 = y[mz];

  sunrealtype h1, h2, h3;
  compute_full_h(y, gx, gy, ng, ny, ncell, m1, m2, m3, &h1, &h2, &h3);

  const sunrealtype mh = m1 * h1 + m2 * h2 + m3 * h3;
  yd[mx] = c_chg * (m3 * h2 - m2 * h3) + c_alpha * (h1 - mh * m1);
  yd[my] = c_chg * (m1 * h3 - m3 * h1) + c_alpha * (h2 - mh * m2);
  yd[mz] = c_chg * (m2 * h1 - m1 * h2) + c_alpha * (h3 - mh * m3);
}

__global__ static void cheap_tiles_kernel(
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

  if (threadIdx.x >= TILE_X || threadIdx.y >= TILE_Y) return;
  if (gx >= ng || gy >= ny) return;

  const int cell = gy * ng + gx;
  const int mx = idx_mx(cell, ncell);
  const int my = idx_my(cell, ncell);
  const int mz = idx_mz(cell, ncell);

  const sunrealtype m1 = y[mx];
  const sunrealtype m2 = y[my];
  const sunrealtype m3 = y[mz];

#if CHEAP_MODE == 1
  yd[mx] = SUN_RCONST(0.0);
  yd[my] = SUN_RCONST(0.0);
  yd[mz] = SUN_RCONST(0.0);
#else
  sunrealtype h1, h2, h3;
  compute_cheap_h(m1, m2, m3, &h1, &h2, &h3);
  const sunrealtype mh = m1 * h1 + m2 * h2 + m3 * h3;
  yd[mx] = c_chg * (m3 * h2 - m2 * h3) + c_alpha * (h1 - mh * m1);
  yd[my] = c_chg * (m1 * h3 - m3 * h1) + c_alpha * (h2 - mh * m2);
  yd[mz] = c_chg * (m2 * h1 - m1 * h2) + c_alpha * (h3 - mh * m3);
#endif
}

static void BuildTileLists(UserData* udata) {
#if REGION_MODE == 0
  udata->ntx = (udata->ng + TILE_X - 1) / TILE_X;
  udata->nty = (udata->ny + TILE_Y - 1) / TILE_Y;
  udata->ntile = udata->ntx * udata->nty;
  udata->n_full_tiles = 0;
  udata->n_cheap_tiles = 0;
  udata->d_full_tile_ix = NULL;
  udata->d_full_tile_iy = NULL;
  udata->d_cheap_tile_ix = NULL;
  udata->d_cheap_tile_iy = NULL;
#else
  udata->ntx = (udata->ng + TILE_X - 1) / TILE_X;
  udata->nty = (udata->ny + TILE_Y - 1) / TILE_Y;
  udata->ntile = udata->ntx * udata->nty;

  int* full_ix = (int*)malloc(sizeof(int) * udata->ntile);
  int* full_iy = (int*)malloc(sizeof(int) * udata->ntile);
  int* cheap_ix = (int*)malloc(sizeof(int) * udata->ntile);
  int* cheap_iy = (int*)malloc(sizeof(int) * udata->ntile);
  if (!full_ix || !full_iy || !cheap_ix || !cheap_iy) {
    fprintf(stderr, "Host tile list allocation failed.\n");
    exit(EXIT_FAILURE);
  }

  const double rr = (double)udata->active_ring_radius;
  const double rw = (double)udata->active_ring_half_width;
  const double slack = (double)udata->tile_safety + (double)HALO_W;
  const double r_outer = rr + rw + slack;
  const double r_inner = fmax(0.0, rr - rw - slack);

  int nf = 0;
  int nc = 0;

  for (int ty = 0; ty < udata->nty; ty++) {
    for (int tx = 0; tx < udata->ntx; tx++) {
      int x0 = tx * TILE_X - HALO_W;
      int y0 = ty * TILE_Y - HALO_W;
      int x1 = tx * TILE_X + TILE_X - 1 + HALO_W;
      int y1 = ty * TILE_Y + TILE_Y - 1 + HALO_W;

      /* conservative sampling of expanded footprint */
      const int px[9] = {
          x0, x1, x0, x1,
          (x0 + x1) / 2, x0, x1,
          (x0 + x1) / 2, (x0 + x1) / 2};
      const int py[9] = {
          y0, y0, y1, y1,
          (y0 + y1) / 2, (y0 + y1) / 2, (y0 + y1) / 2,
          y0, y1};

      double minrho = 1.0e100;
      double maxrho = -1.0e100;
      for (int k = 0; k < 9; k++) {
        int sx = wrap_x(px[k], udata->ng);
        int sy = wrap_y(py[k], udata->ny);
        const double dx = (double)sx - (double)udata->cx;
        const double dy = (double)sy - (double)udata->cy;
        const double rho = sqrt(dx * dx + dy * dy);
        if (rho < minrho) minrho = rho;
        if (rho > maxrho) maxrho = rho;
      }

      const int intersects = (maxrho >= r_inner && minrho <= r_outer);
      if (intersects) {
        full_ix[nf] = tx;
        full_iy[nf] = ty;
        nf++;
      } else {
        cheap_ix[nc] = tx;
        cheap_iy[nc] = ty;
        nc++;
      }
    }
  }

  udata->n_full_tiles = nf;
  udata->n_cheap_tiles = nc;

  if (nf > 0) {
    CHECK_CUDA(cudaMalloc((void**)&udata->d_full_tile_ix, sizeof(int) * nf));
    CHECK_CUDA(cudaMalloc((void**)&udata->d_full_tile_iy, sizeof(int) * nf));
    CHECK_CUDA(cudaMemcpy(udata->d_full_tile_ix, full_ix, sizeof(int) * nf, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(udata->d_full_tile_iy, full_iy, sizeof(int) * nf, cudaMemcpyHostToDevice));
  } else {
    udata->d_full_tile_ix = NULL;
    udata->d_full_tile_iy = NULL;
  }

  if (nc > 0) {
    CHECK_CUDA(cudaMalloc((void**)&udata->d_cheap_tile_ix, sizeof(int) * nc));
    CHECK_CUDA(cudaMalloc((void**)&udata->d_cheap_tile_iy, sizeof(int) * nc));
    CHECK_CUDA(cudaMemcpy(udata->d_cheap_tile_ix, cheap_ix, sizeof(int) * nc, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(udata->d_cheap_tile_iy, cheap_iy, sizeof(int) * nc, cudaMemcpyHostToDevice));
  } else {
    udata->d_cheap_tile_ix = NULL;
    udata->d_cheap_tile_iy = NULL;
  }

  free(full_ix);
  free(full_iy);
  free(cheap_ix);
  free(cheap_iy);
#endif
}

static void FreeTileLists(UserData* udata) {
  if (udata->d_full_tile_ix) CHECK_CUDA(cudaFree(udata->d_full_tile_ix));
  if (udata->d_full_tile_iy) CHECK_CUDA(cudaFree(udata->d_full_tile_iy));
  if (udata->d_cheap_tile_ix) CHECK_CUDA(cudaFree(udata->d_cheap_tile_ix));
  if (udata->d_cheap_tile_iy) CHECK_CUDA(cudaFree(udata->d_cheap_tile_iy));
  udata->d_full_tile_ix = NULL;
  udata->d_full_tile_iy = NULL;
  udata->d_cheap_tile_ix = NULL;
  udata->d_cheap_tile_iy = NULL;
}

static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
  (void)t;

  UserData* udata = (UserData*)user_data;
  sunrealtype* ydata    = N_VGetDeviceArrayPointer_Cuda(y);
  sunrealtype* ydotdata = N_VGetDeviceArrayPointer_Cuda(ydot);

  if (BLOCK_X * BLOCK_Y > 1024) {
    fprintf(stderr, "Invalid block size: BLOCK_X * BLOCK_Y = %d > 1024\n",
            BLOCK_X * BLOCK_Y);
    return -1;
  }
  if (TILE_X != BLOCK_X || TILE_Y != BLOCK_Y) {
    fprintf(stderr, "For this version require TILE_X==BLOCK_X and TILE_Y==BLOCK_Y.\n");
    return -1;
  }

#if REGION_MODE == 0
  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid((udata->ng + block.x - 1) / block.x,
            (udata->ny + block.y - 1) / block.y);
  full_kernel_uniform<<<grid, block>>>(ydata, ydotdata, udata->ng, udata->ny, udata->ncell);
#else
  dim3 block(TILE_X, TILE_Y);
  if (udata->n_full_tiles > 0) {
    dim3 grid_full((unsigned int)udata->n_full_tiles, 1, 1);
    full_tiles_kernel<<<grid_full, block>>>(
        ydata, ydotdata, udata->ng, udata->ny, udata->ncell,
        udata->d_full_tile_ix, udata->d_full_tile_iy, udata->n_full_tiles);
  }
  if (udata->n_cheap_tiles > 0) {
    dim3 grid_cheap((unsigned int)udata->n_cheap_tiles, 1, 1);
    cheap_tiles_kernel<<<grid_cheap, block>>>(
        ydata, ydotdata, udata->ng, udata->ny, udata->ncell,
        udata->d_cheap_tile_ix, udata->d_cheap_tile_iy, udata->n_cheap_tiles);
  }
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
  udata.cx    = cx;
  udata.cy    = cy;
  udata.active_ring_radius = SUN_RCONST(ACTIVE_RING_RADIUS_FRAC_Y) * (sunrealtype)ny;
  udata.active_ring_half_width = SUN_RCONST(ACTIVE_RING_HALF_WIDTH_FRAC_Y) * (sunrealtype)ny;
  udata.tile_safety = SUN_RCONST(TILE_SAFETY_FRAC_Y) * (sunrealtype)ny;

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
  BuildTileLists(&udata);

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
  printf("circle center                 = (%.2f, %.2f)\n", (double)cx, (double)cy);
  printf("initial texture radius        = %.2f cells\n", (double)radius);
  printf("smooth radial texture init\n");
  printf("core mz                       = %.4f\n", (double)TEXTURE_CORE_MZ);
  printf("outer mz                      = %.4f\n", (double)TEXTURE_OUTER_MZ);
  printf("width frac                    = %.4f\n", (double)TEXTURE_WIDTH_FRAC);
  printf("T_TOTAL                       = %.2f\n", (double)T_TOTAL);
  printf("REGION_MODE                   = %d\n", REGION_MODE);
  printf("ACTIVE_RING_RADIUS_FRAC_Y     = %.4f\n", (double)ACTIVE_RING_RADIUS_FRAC_Y);
  printf("ACTIVE_RING_HALF_WIDTH_FRAC_Y = %.4f\n", (double)ACTIVE_RING_HALF_WIDTH_FRAC_Y);
  printf("CHEAP_MODE                    = %d\n", CHEAP_MODE);
  printf("HALO_W                        = %d\n", HALO_W);
  printf("TILE_SAFETY_FRAC_Y            = %.4f\n", (double)TILE_SAFETY_FRAC_Y);
  printf("tile size                     = %d x %d\n", TILE_X, TILE_Y);
#if REGION_MODE == 1
  printf("tiles: total=%d, full=%d, cheap=%d\n", udata.ntile, udata.n_full_tiles, udata.n_cheap_tiles);
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
    if (ShouldWriteFrame(iout + 1, t)) WriteFrame(fp, t, nx, ny, ng, ncell, y);
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
  FreeTileLists(&udata);
  if (LS) SUNLinSolFree(LS);
  if (NLS) SUNNonlinSolFree(NLS);
  if (cvode_mem) CVodeFree(&cvode_mem);
  if (y) N_VDestroy(y);
  if (abstol) N_VDestroy(abstol);
  if (sunctx) SUNContext_Free(&sunctx);
#if ENABLE_OUTPUT
  fclose(fp);
#endif
  return 0;
}
