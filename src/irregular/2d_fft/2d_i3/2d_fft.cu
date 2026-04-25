/**
 * 2D LLG solver — irregular antidot mesh + soft-edge mask + FFT demag
 * Drop-in replacement for src/irregular/2d_fft/2d_i1/2d_fft.cu
 *
 * ─── Geometry ────────────────────────────────────────────────────────
 * N_HOLES_X × N_HOLES_Y stratified-jitter grid of ellipse holes:
 *   - center jittered within each super-cell
 *   - random aspect ratio in [HOLE_ASPECT_MIN, HOLE_ASPECT_MAX]
 *   - random rotation in [0, π]
 *   - periodic copies (3×3 tile) included so PBC tiles cleanly
 *
 * Continuous weight field on host (computed once):
 *   w(x,y) = 0.5 · (1 + tanh(SDF(x,y) / MASK_EPS_CELLS))
 *
 * ─── Physics ─────────────────────────────────────────────────────────
 *   - exchange / DMI use weighted neighbors  w_j · n̂_j   (via y_eff buffer)
 *   - anisotropy uses self n̂  (unit vector after normalize)
 *   - demag uses w · n̂ globally — long-range field naturally tapered at holes
 *   - LLG governs n̂ on unit sphere; output yd = LLG(...) · w[self]
 *     so deep-hole cells stay frozen and boundary cells evolve fractionally
 *   - normalize_m_kernel: only re-projects cells with w > 1e-3
 *
 * ─── PMPP optimizations ──────────────────────────────────────────────
 *   - DENSE execution, branch-free hot path (no `if active` in RHS kernel)
 *   - all reads coalesced (SoA + dense launch); w is 1D float-double per cell
 *   - one fused kernel: exchange + aniso + DMI + demag + LLG + w-scale
 *   - pre-pass `apply_weight_kernel` writes y_eff = w·y once, reused by
 *     both Demag_Apply (gather) and the unified RHS (neighbor sums)
 *   - one extra normalize check (compare w to 1e-3) — single FMA, no branch
 *     divergence within a warp because the comparison is on read-only data
 *
 * Build:  make run-demag PRINT=1
 * Plot:   mdyn2D.m on output.txt (arrows fade smoothly to 0 in holes)
 *         mask.txt is also written once for overlay if desired
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

/* ─── Solver tuning ──────────────────────────────────────────────── */
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

/* ─── Demag controls ─────────────────────────────────────────────── */
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

/* ─── Initial-condition knobs (head-on three-stripe) ─────────────── */
#ifndef STRIPE_LEFT_FRAC
#define STRIPE_LEFT_FRAC 0.25
#endif
#ifndef STRIPE_RIGHT_FRAC
#define STRIPE_RIGHT_FRAC 0.75
#endif
#ifndef INIT_RANDOM_EPS
#define INIT_RANDOM_EPS 0.01
#endif
#ifndef INIT_RANDOM_SEED
#define INIT_RANDOM_SEED 12345
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

/* ─── Irregular antidot-mesh geometry knobs ──────────────────────── */
#ifndef N_HOLES_X
#define N_HOLES_X 4
#endif
#ifndef N_HOLES_Y
#define N_HOLES_Y 4
#endif
#define N_HOLES (N_HOLES_X * N_HOLES_Y)

#ifndef HOLE_SEED
#define HOLE_SEED 20251101
#endif
#ifndef MASK_EPS_CELLS
#define MASK_EPS_CELLS 1.5      /* tanh transition half-width, in cells   */
#endif
#ifndef HOLE_BASE_FRAC
#define HOLE_BASE_FRAC 0.18     /* base radius as fraction of super-cell  */
#endif
#ifndef HOLE_GROWTH_FRAC
#define HOLE_GROWTH_FRAC 0.14   /* extra random radius up to this         */
#endif
#ifndef HOLE_ASPECT_MIN
#define HOLE_ASPECT_MIN 0.55
#endif
#ifndef HOLE_ASPECT_MAX
#define HOLE_ASPECT_MAX 1.85
#endif
#ifndef HOLE_JITTER
#define HOLE_JITTER 0.60        /* center stays in central HOLE_JITTER fraction */
#endif

#ifndef GRID_NX
#define GRID_NX 768             /* total scalar width = GROUPSIZE * ng */
#endif
#ifndef GRID_NY
#define GRID_NY 256
#endif

/* ─── Material constants (device constant memory) ────────────────── */
__constant__ sunrealtype c_msk[3] = {SUN_RCONST(1.0), SUN_RCONST(0.0), SUN_RCONST(0.0)};
__constant__ sunrealtype c_nsk[3] = {SUN_RCONST(1.0), SUN_RCONST(0.0), SUN_RCONST(0.0)};
__constant__ sunrealtype c_chk   = SUN_RCONST(4.0);
__constant__ sunrealtype c_che   = SUN_RCONST(4.0);
__constant__ sunrealtype c_alpha = SUN_RCONST(0.2);
__constant__ sunrealtype c_chg   = SUN_RCONST(1.0);
__constant__ sunrealtype c_cha   = SUN_RCONST(0.0);
__constant__ sunrealtype c_chb   = SUN_RCONST(0.3);

/* ─── Error checking macros ──────────────────────────────────────── */
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

/* ─── UserData ───────────────────────────────────────────────────── */
/* Layout-critical first 16 bytes (pd, demag, d_hdmag) and offsets up
 * through nzz0 (offset 64) MUST match precond.cu / jtv.cu mirrors.
 * The new fields (d_w, d_y_eff, h_w) sit AFTER nzz0 and are invisible
 * to the precond/jtv casts, so no header churn there. */
typedef struct {
  PrecondData  *pd;              /* offset 0  */
  DemagData    *demag;           /* offset 8  */
  sunrealtype  *d_hdmag;         /* offset 16 */
  int nx;                        /* offset 24 */
  int ny;                        /* offset 28 */
  int ng;                        /* offset 32 */
  int ncell;                     /* offset 36 */
  int neq;                       /* offset 40 */
  /* 4 bytes pad before doubles */
  double nxx0;                   /* offset 48 */
  double nyy0;                   /* offset 56 */
  double nzz0;                   /* offset 64 */
  /* === extensions (after the precond/jtv mirrors) === */
  sunrealtype  *d_w;             /* per-cell soft weight, device */
  sunrealtype  *d_y_eff;         /* scratch w·y, device, 3*ncell  */
  double       *h_w;             /* host copy of weight (for I/O) */
} UserData;

/* ─── SoA indexing helpers ───────────────────────────────────────── */
__host__ __device__ static inline int idx_mx(int cell, int ncell) { return cell; }
__host__ __device__ static inline int idx_my(int cell, int ncell) { return ncell + cell; }
__host__ __device__ static inline int idx_mz(int cell, int ncell) { return 2*ncell + cell; }
__host__ __device__ static inline int wrap_x(int x, int ng) {
  return (x < 0) ? (x + ng) : ((x >= ng) ? (x - ng) : x);
}
__host__ __device__ static inline int wrap_y(int y, int ny) {
  return (y < 0) ? (y + ny) : ((y >= ny) ? (y - ny) : y);
}

/* ─── Hole geometry (host-side) ──────────────────────────────────── */
typedef struct {
  double cx, cy;       /* center in cell coords */
  double rx, ry;       /* semi-axes in cells    */
  double cos_t, sin_t; /* rotation              */
} HoleSpec;

static HoleSpec g_holes[N_HOLES];

static double rand_unit(void) {
  return (double)rand() / (double)RAND_MAX;
}

static void generate_holes(int ng, int ny) {
  srand((unsigned)HOLE_SEED);
  const double dx = (double)ng / (double)N_HOLES_X;
  const double dy = (double)ny / (double)N_HOLES_Y;
  const double mean_super = 0.5 * (dx + dy);
  int idx = 0;
  for (int j = 0; j < N_HOLES_Y; j++) {
    for (int i = 0; i < N_HOLES_X; i++) {
      const double margin = (1.0 - HOLE_JITTER) * 0.5;
      const double cx = ((double)i + margin + HOLE_JITTER * rand_unit()) * dx;
      const double cy = ((double)j + margin + HOLE_JITTER * rand_unit()) * dy;
      const double base = HOLE_BASE_FRAC + HOLE_GROWTH_FRAC * rand_unit();
      const double mean_r = base * mean_super;
      const double aspect = HOLE_ASPECT_MIN
        + (HOLE_ASPECT_MAX - HOLE_ASPECT_MIN) * rand_unit();
      const double r1 = mean_r * sqrt(aspect);
      const double r2 = mean_r / sqrt(aspect);
      const double theta = M_PI * rand_unit();
      g_holes[idx].cx = cx;
      g_holes[idx].cy = cy;
      g_holes[idx].rx = r1;
      g_holes[idx].ry = r2;
      g_holes[idx].cos_t = cos(theta);
      g_holes[idx].sin_t = sin(theta);
      idx++;
    }
  }
}

/* Signed distance to the union of 16 holes + their 8 periodic copies.
 * Positive outside all holes, negative inside any. */
static double sdf_at(double x, double y, int ng, int ny) {
  double best = 1.0e30;
  for (int dy_p = -1; dy_p <= 1; dy_p++) {
    for (int dx_p = -1; dx_p <= 1; dx_p++) {
      for (int h = 0; h < N_HOLES; h++) {
        const double cx = g_holes[h].cx + (double)dx_p * (double)ng;
        const double cy = g_holes[h].cy + (double)dy_p * (double)ny;
        const double rx = x - cx;
        const double ry = y - cy;
        const double lx =  rx * g_holes[h].cos_t + ry * g_holes[h].sin_t;
        const double ly = -rx * g_holes[h].sin_t + ry * g_holes[h].cos_t;
        const double a = g_holes[h].rx;
        const double b = g_holes[h].ry;
        const double rnorm = sqrt((lx/a)*(lx/a) + (ly/b)*(ly/b));
        /* convert normalized ellipse coordinate to physical-ish distance
         * by multiplying by min semi-axis (good enough for soft tanh). */
        const double sdf_h = (rnorm - 1.0) * fmin(a, b);
        if (sdf_h < best) best = sdf_h;
      }
    }
  }
  return best;
}

static void build_weight(int ng, int ny, double *h_w) {
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < ng; i++) {
      const double sdf = sdf_at((double)i + 0.5, (double)j + 0.5, ng, ny);
      double w = 0.5 * (1.0 + tanh(sdf / (double)MASK_EPS_CELLS));
      if (w < 1.0e-3) w = 0.0;
      if (w > 1.0 - 1.0e-3) w = 1.0;
      h_w[j * ng + i] = w;
    }
  }
}

static void write_mask_file(int ng, int ny, const double *h_w) {
  FILE *fm = fopen("mask.txt", "w");
  if (!fm) return;
  fprintf(fm, "%d %d\n", ng, ny);
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < ng; i++) {
      fprintf(fm, "%f\n", h_w[j*ng + i]);
    }
  }
  fclose(fm);
}

/* ─── Kernel: y_eff = w · y  (one pass, coalesced) ───────────────── */
__global__ static void apply_weight_kernel(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ w,
    sunrealtype*       __restrict__ y_eff,
    int ncell)
{
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= ncell) return;
  const sunrealtype ww = w[cell];
  y_eff[cell]              = ww * y[cell];
  y_eff[ncell + cell]      = ww * y[ncell + cell];
  y_eff[2 * ncell + cell]  = ww * y[2 * ncell + cell];
}

/* ─── Kernel: in-place |m|=1 normalization on active cells only ──── */
__global__ static void normalize_m_kernel_w(
    sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ w,
    int ncell)
{
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= ncell) return;
  if (w[cell] < SUN_RCONST(1.0e-3)) return;   /* skip deep-hole cells */

  const int mx = idx_mx(cell, ncell);
  const int my = idx_my(cell, ncell);
  const int mz = idx_mz(cell, ncell);

  const sunrealtype m1 = y[mx];
  const sunrealtype m2 = y[my];
  const sunrealtype m3 = y[mz];
  const sunrealtype ymp = sqrt(m1*m1 + m2*m2 + m3*m3);
  if (ymp > SUN_RCONST(1.0e-30)) {
    const sunrealtype inv = SUN_RCONST(1.0) / ymp;
    y[mx] = m1 * inv;
    y[my] = m2 * inv;
    y[mz] = m3 * inv;
  }
}

/* ─── Unified RHS kernel — soft-irregular dense, branch-free ─────── */
/*
 * Reads:
 *   y[self]                 — unit vector (after normalize)
 *   y_eff[neighbors]        — w·n̂ at the 4 neighbors, used for
 *                             exchange and DMI sums
 *   w[self]                 — for final yd scaling
 *   h_dmag[self]            — demag field from FFT( w·n̂ )
 *
 * Writes:
 *   yd = w[self] · LLG(n̂_self, h_eff)
 *
 * No branch on hole/active.  Cells with w_self ≈ 0 produce yd ≈ 0
 * automatically; cells with w_self ∈ (0,1) evolve at fractional rate.
 */
__global__ static void f_kernel_unified_irregular_soft(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ y_eff,
    const sunrealtype* __restrict__ w,
    const sunrealtype* __restrict__ h_dmag,
    sunrealtype*       __restrict__ yd,
    int ng, int ny, int ncell)
{
  const int gx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gy = blockIdx.y * blockDim.y + threadIdx.y;
  if (gx >= ng || gy >= ny) return;

  const int cell = gy * ng + gx;
  const int mx_i = idx_mx(cell, ncell);
  const int my_i = idx_my(cell, ncell);
  const int mz_i = idx_mz(cell, ncell);

  const int xl   = wrap_x(gx - 1, ng);
  const int xr   = wrap_x(gx + 1, ng);
  const int yu   = wrap_y(gy - 1, ny);
  const int ydn  = wrap_y(gy + 1, ny);

  const int lc = gy  * ng + xl;
  const int rc = gy  * ng + xr;
  const int uc = yu  * ng + gx;
  const int dc = ydn * ng + gx;

  const sunrealtype m1 = y[mx_i];
  const sunrealtype m2 = y[my_i];
  const sunrealtype m3 = y[mz_i];
  const sunrealtype w_self = w[cell];

  /* Effective neighbor m's (already w·n̂) — coalesced reads. */
  const sunrealtype lx1 = y_eff[idx_mx(lc, ncell)];
  const sunrealtype lx2 = y_eff[idx_my(lc, ncell)];
  const sunrealtype lx3 = y_eff[idx_mz(lc, ncell)];
  const sunrealtype rx1 = y_eff[idx_mx(rc, ncell)];
  const sunrealtype rx2 = y_eff[idx_my(rc, ncell)];
  const sunrealtype rx3 = y_eff[idx_mz(rc, ncell)];
  const sunrealtype ux1 = y_eff[idx_mx(uc, ncell)];
  const sunrealtype ux2 = y_eff[idx_my(uc, ncell)];
  const sunrealtype ux3 = y_eff[idx_mz(uc, ncell)];
  const sunrealtype dx1 = y_eff[idx_mx(dc, ncell)];
  const sunrealtype dx2 = y_eff[idx_my(dc, ncell)];
  const sunrealtype dx3 = y_eff[idx_mz(dc, ncell)];

  /* h_eff = exchange(weighted) + anisotropy(self) + DMI(weighted x-nbrs) + demag */
  const sunrealtype h1 =
      c_che * (lx1 + rx1 + ux1 + dx1)
    + c_msk[0] * (c_chk * m1 + c_cha)
    + c_chb * c_nsk[0] * (lx1 + rx1)
    + h_dmag[mx_i];
  const sunrealtype h2 =
      c_che * (lx2 + rx2 + ux2 + dx2)
    + c_msk[1] * (c_chk * m1 + c_cha)
    + c_chb * c_nsk[1] * (lx2 + rx2)
    + h_dmag[my_i];
  const sunrealtype h3 =
      c_che * (lx3 + rx3 + ux3 + dx3)
    + c_msk[2] * (c_chk * m1 + c_cha)
    + c_chb * c_nsk[2] * (lx3 + rx3)
    + h_dmag[mz_i];

  const sunrealtype mh = m1 * h1 + m2 * h2 + m3 * h3;

  /* Standard LLG for n̂ (|n̂|=1 enforced by normalize_m_kernel_w),
   * scaled by w_self for soft boundary dynamics. */
  yd[mx_i] = w_self * (c_chg * (m3 * h2 - m2 * h3) + c_alpha * (h1 - mh * m1));
  yd[my_i] = w_self * (c_chg * (m1 * h3 - m3 * h1) + c_alpha * (h2 - mh * m2));
  yd[mz_i] = w_self * (c_chg * (m2 * h1 - m1 * h2) + c_alpha * (h3 - mh * m3));
}

/* ─── RHS wrapper for CVODE ──────────────────────────────────────── */
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

  /* Step 0: |n̂|=1 in active cells (skip deep-hole). */
  {
    const int nb = 256;
    const int gd = (udata->ncell + nb - 1) / nb;
    normalize_m_kernel_w<<<gd, nb>>>(ydata, udata->d_w, udata->ncell);
  }

  /* Step 1: y_eff = w · y  (used by both demag gather and unified RHS). */
  {
    const int nb = 256;
    const int gd = (udata->ncell + nb - 1) / nb;
    apply_weight_kernel<<<gd, nb>>>(ydata, udata->d_w, udata->d_y_eff,
                                    udata->ncell);
  }

  /* Step 2: demag — feed Demag_Apply the windowed magnetization. */
  if (udata->demag && DEMAG_STRENGTH > 0.0) {
    Demag_Apply(udata->demag,
                (const double*)udata->d_y_eff,
                (double*)udata->d_hdmag);
  } else {
    cudaMemsetAsync(udata->d_hdmag, 0,
                    (size_t)3 * udata->ncell * sizeof(sunrealtype), 0);
  }

  /* Step 3: unified RHS — fuse exchange + aniso + DMI + demag + LLG + w */
  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid((udata->ng + block.x - 1) / block.x,
            (udata->ny + block.y - 1) / block.y);

  f_kernel_unified_irregular_soft<<<grid, block>>>(
      ydata, udata->d_y_eff, udata->d_w,
      udata->d_hdmag, ydotdata,
      udata->ng, udata->ny, udata->ncell);

  cudaError_t cuerr = cudaPeekAtLastError();
  if (cuerr != cudaSuccess) {
    fprintf(stderr, ">>> ERROR in f: kernel launch failed: %s\n",
            cudaGetErrorString(cuerr));
    return -1;
  }
  return 0;
}

/* ─── Final-stats reporter ───────────────────────────────────────── */
static void PrintFinalStats(void* cvode_mem, SUNLinearSolver LS) {
  (void)LS;
  long int nst, nfe, nsetups, nni, ncfn, netf, nge, nli, nlcf, njvevals;
  CVodeGetNumSteps(cvode_mem, &nst);
  CVodeGetNumRhsEvals(cvode_mem, &nfe);
  CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
  CVodeGetNumErrTestFails(cvode_mem, &netf);
  CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
  CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
  CVodeGetNumGEvals(cvode_mem, &nge);
  CVodeGetNumLinIters(cvode_mem, &nli);
  CVodeGetNumLinConvFails(cvode_mem, &nlcf);
  CVodeGetNumJtimesEvals(cvode_mem, &njvevals);
  printf("\nFinal Statistics:\n");
  printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld ", nst, nfe, nsetups);
  printf("nni = %-6ld ncfn = %-6ld netf = %-6ld nge = %ld\n",
         nni, ncfn, netf, nge);
  printf("nli = %-6ld nlcf = %-6ld njvevals = %ld\n", nli, nlcf, njvevals);
}

#if ENABLE_OUTPUT
/* WriteFrame: outputs m·w_soft so quiver naturally fades to 0 in holes,
 * giving the mesh-like "no obvious grid" visual without any matlab edits. */
static void WriteFrame(FILE* fp,
                       sunrealtype t,
                       int nx, int ny, int ng, int ncell,
                       N_Vector y,
                       const double *h_w)
{
  N_VCopyFromDevice_Cuda(y);
  sunrealtype* ydata = N_VGetHostArrayPointer_Cuda(y);

  fprintf(fp, "%f %d %d\n", (double)t, nx, ny);
  for (int jp = 0; jp < ny; jp++) {
    for (int ip = 0; ip < ng; ip++) {
      const int cell_out = jp * ng + ip;
      const double w = h_w[cell_out];
      fprintf(fp, "%f %f %f\n",
              w * (double)ydata[idx_mx(cell_out, ncell)],
              w * (double)ydata[idx_my(cell_out, ncell)],
              w * (double)ydata[idx_mz(cell_out, ncell)]);
    }
  }
  fprintf(fp, "\n");
}

static int ShouldWriteFrame(long int iout, sunrealtype t) {
  if (t <= SUN_RCONST(EARLY_SAVE_UNTIL)) return (iout % EARLY_SAVE_EVERY) == 0;
  return (iout % LATE_SAVE_EVERY) == 0;
}
#endif

/* ─── main ───────────────────────────────────────────────────────── */
int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

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

  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;

  const int nx = GRID_NX;
  const int ny = GRID_NY;

  if (nx % GROUPSIZE != 0) {
    fprintf(stderr, "nx must be a multiple of GROUPSIZE=%d\n", GROUPSIZE);
    return 1;
  }
  const int ng    = nx / GROUPSIZE;
  const int ncell = ng * ny;
  const int neq   = 3 * ncell;

#if ENABLE_OUTPUT
  FILE* fp = fopen("output.txt", "w");
  if (!fp) { fprintf(stderr, "Error opening output file.\n"); return 1; }
  setvbuf(fp, NULL, _IOFBF, 1 << 20);
#else
  FILE* fp = NULL; (void)fp;
#endif

  const int q1 = (int)(STRIPE_LEFT_FRAC  * (double)ng + 0.5);
  const int q3 = (int)(STRIPE_RIGHT_FRAC * (double)ng + 0.5);

  udata.nx = nx; udata.ny = ny; udata.ng = ng;
  udata.ncell = ncell; udata.neq = neq;

  CHECK_SUNDIALS(SUNContext_Create(SUN_COMM_NULL, &sunctx));

  /* ─── Build geometry: 16 holes + soft weight on host ──────────── */
  udata.h_w = (double*)malloc((size_t)ncell * sizeof(double));
  if (!udata.h_w) { fprintf(stderr, "h_w alloc failed\n"); return 1; }
  generate_holes(ng, ny);
  build_weight(ng, ny, udata.h_w);

  /* Stats for the log line */
  {
    double w_sum = 0.0;
    int    n_full = 0, n_hole = 0, n_boundary = 0;
    for (int k = 0; k < ncell; k++) {
      const double w = udata.h_w[k];
      w_sum += w;
      if (w <= 1.0e-3)         n_hole++;
      else if (w >= 1.0-1.0e-3) n_full++;
      else                       n_boundary++;
    }
    printf("[Geometry] %d holes (%dx%d stratified-jitter), seed=%d, eps=%.2f cells\n",
           N_HOLES, N_HOLES_X, N_HOLES_Y, (int)HOLE_SEED, (double)MASK_EPS_CELLS);
    printf("[Geometry] cells: %d total, %d full (w=1), %d boundary (0<w<1), %d deep-hole (w=0)\n",
           ncell, n_full, n_boundary, n_hole);
    printf("[Geometry] effective active fraction (sum w / N) = %.4f\n",
           w_sum / (double)ncell);
  }

  write_mask_file(ng, ny, udata.h_w);

  /* upload weight + alloc y_eff scratch */
  CHECK_CUDA(cudaMalloc((void**)&udata.d_w, (size_t)ncell * sizeof(sunrealtype)));
  {
    /* h_w is double; sunrealtype is also double — direct copy. */
    CHECK_CUDA(cudaMemcpy(udata.d_w, udata.h_w,
                          (size_t)ncell * sizeof(sunrealtype),
                          cudaMemcpyHostToDevice));
  }
  CHECK_CUDA(cudaMalloc((void**)&udata.d_y_eff,
                        (size_t)3 * ncell * sizeof(sunrealtype)));
  CHECK_CUDA(cudaMemset(udata.d_y_eff, 0,
                        (size_t)3 * ncell * sizeof(sunrealtype)));

  /* 3x3 block-Jacobi preconditioner (unchanged) */
  udata.pd = Precond_Create(ng, ny, ncell);
  if (!udata.pd) { fprintf(stderr, "Precond_Create failed\n"); return 1; }

  /* d_hdmag always allocated (zero-fill if demag off) */
  CHECK_CUDA(cudaMalloc((void**)&udata.d_hdmag,
                        (size_t)3 * ncell * sizeof(sunrealtype)));
  CHECK_CUDA(cudaMemset(udata.d_hdmag, 0,
                        (size_t)3 * ncell * sizeof(sunrealtype)));

  /* FFT demag init (Newell tensor f̂) */
  const double dstr = (double)DEMAG_STRENGTH;
  const double dthk = (double)DEMAG_THICK;
  if (dstr > 0.0) {
    udata.demag = Demag_Init(ng, ny, dthk, dstr);
    if (!udata.demag) {
      fprintf(stderr, "Demag_Init failed\n");
      Precond_Destroy(udata.pd);
      cudaFree(udata.d_hdmag);
      cudaFree(udata.d_w);
      cudaFree(udata.d_y_eff);
      return 1;
    }
    Demag_GetSelfCoupling(udata.demag,
                          &udata.nxx0, &udata.nyy0, &udata.nzz0);
    printf("[main] Demag self-coupling (strength-scaled): "
           "nxx0=%.4e  nyy0=%.4e  nzz0=%.4e\n",
           udata.nxx0, udata.nyy0, udata.nzz0);
  }

  y      = N_VNew_Cuda(neq, sunctx);
  abstol = N_VNew_Cuda(neq, sunctx);
  if (!y || !abstol) { fprintf(stderr, "N_Vector alloc failed\n"); goto cleanup; }
  FusedNVec_Init(y);

  ydata       = N_VGetHostArrayPointer_Cuda(y);
  abstol_data = N_VGetHostArrayPointer_Cuda(abstol);

  /* ─── Initial condition: head-on stripes, masked by w ─────────── */
  {
    const double eps = (double)INIT_RANDOM_EPS;
    srand((unsigned)INIT_RANDOM_SEED);
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < ng; i++) {
        const int cell = j * ng + i;
        const int mx_i = idx_mx(cell, ncell);
        const int my_i = idx_my(cell, ncell);
        const int mz_i = idx_mz(cell, ncell);

        if (udata.h_w[cell] > 1.0e-3) {
          double mx0 = (i >= q1 && i < q3) ? 1.0 : -1.0;
          double my0 = eps * (2.0 * (double)rand()/(double)RAND_MAX - 1.0);
          double mz0 = eps * (2.0 * (double)rand()/(double)RAND_MAX - 1.0);
          const double n = sqrt(mx0*mx0 + my0*my0 + mz0*mz0);
          mx0 /= n; my0 /= n; mz0 /= n;
          ydata[mx_i] = SUN_RCONST(mx0);
          ydata[my_i] = SUN_RCONST(my0);
          ydata[mz_i] = SUN_RCONST(mz0);
        } else {
          ydata[mx_i] = ZERO;
          ydata[my_i] = ZERO;
          ydata[mz_i] = ZERO;
        }
        abstol_data[mx_i] = ATOL1;
        abstol_data[my_i] = ATOL2;
        abstol_data[mz_i] = ATOL3;
      }
    }
  }
  N_VCopyToDevice_Cuda(y);
  N_VCopyToDevice_Cuda(abstol);

#if ENABLE_OUTPUT
  WriteFrame(fp, T0, nx, ny, ng, ncell, y, udata.h_w);
#endif

  cvode_mem = CVodeCreate(CV_BDF, sunctx);
  if (!cvode_mem) { fprintf(stderr, "CVodeCreate failed.\n"); goto cleanup; }

  CHECK_SUNDIALS(CVodeInit(cvode_mem, f, T0, y));
  CHECK_SUNDIALS(CVodeSetUserData(cvode_mem, &udata));
  CHECK_SUNDIALS(CVodeSVtolerances(cvode_mem, RTOL, abstol));

  NLS = SUNNonlinSol_Newton(y, sunctx);
  if (!NLS) { fprintf(stderr, "SUNNonlinSol_Newton failed.\n"); goto cleanup; }
  CHECK_SUNDIALS(CVodeSetNonlinearSolver(cvode_mem, NLS));

  LS = SUNLinSol_SPGMR(y, SUN_PREC_LEFT, KRYLOV_DIM, sunctx);
  if (!LS) { fprintf(stderr, "SUNLinSol_SPGMR failed.\n"); goto cleanup; }
  CHECK_SUNDIALS(CVodeSetLinearSolver(cvode_mem, LS, NULL));
  CHECK_SUNDIALS(CVodeSetJacTimes(cvode_mem, NULL, JtvProduct));
  CHECK_SUNDIALS(CVodeSetPreconditioner(cvode_mem, PrecondSetup, PrecondSolve));

  if (neq < 500000) {
    CHECK_SUNDIALS(SUNLinSol_SPGMRSetGSType(LS, SUN_CLASSICAL_GS));
    printf("GS type: Classical (overhead-limited, neq=%d)\n", neq);
  } else {
    printf("GS type: Modified  (bandwidth-limited, neq=%d)\n", neq);
  }
  CHECK_SUNDIALS(CVodeSetMaxOrd(cvode_mem, MAX_BDF_ORDER));

  printf("\n2D irregular antidot mesh + soft-edge mask + FFT demag\n");
  printf("LLG form: standard, |n̂|=1 enforced by normalize-in-f, yd scaled by w\n\n");
  printf("nx=%d  ny=%d  ng=%d  ncell=%d  neq=%d\n", nx, ny, ng, ncell, neq);
  printf("periodic BC: x and y\n");
  printf("Init: head-on stripes  q1=%d (x=%.2f*ng)  q3=%d (x=%.2f*ng)\n",
         q1, (double)STRIPE_LEFT_FRAC, q3, (double)STRIPE_RIGHT_FRAC);
  printf("DEMAG_STRENGTH=%.4f  DEMAG_THICK=%.4f  (%s)\n",
         dstr, dthk,
         dstr > 0.0 ? "h_dmag = IFFT[f_hat * (w·m)_hat]"
                    : "demag disabled");
  printf("T_TOTAL=%.2f  RTOL/ATOL=%.1e\n",
         (double)T_TOTAL, (double)RTOL_VAL);

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
    if (ShouldWriteFrame(iout + 1, t))
      WriteFrame(fp, t, nx, ny, ng, ncell, y, udata.h_w);
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
  if (LS) SUNLinSolFree(LS);
  if (NLS) SUNNonlinSolFree(NLS);
  if (cvode_mem) CVodeFree(&cvode_mem);
  if (y) N_VDestroy(y);
  if (abstol) N_VDestroy(abstol);
  if (sunctx) SUNContext_Free(&sunctx);
  Precond_Destroy(udata.pd);
  Demag_Destroy(udata.demag);
  if (udata.d_hdmag) cudaFree(udata.d_hdmag);
  if (udata.d_w)     cudaFree(udata.d_w);
  if (udata.d_y_eff) cudaFree(udata.d_y_eff);
  if (udata.h_w)     free(udata.h_w);
  FusedNVec_FreePool();
#if ENABLE_OUTPUT
  if (fp) fclose(fp);
#endif
  return 0;
}