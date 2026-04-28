/**
 * 2D irregular LLG solver — ELLIPTICAL active geometry + COMPACT active-cell
 * execution + FFT demag.  CVODE + CUDA, SoA layout.
 *
 * Derived from 2d_i5 (centered-rectangle geometry, ymsk-mask execution)
 * by adopting 2d_i1's compact active-cell execution model.
 *
 * ─── Geometry (i6) ──────────────────────────────────────────────────
 * Centered ELLIPSE with semi-axes
 *   rx = ACTIVE_RX_FRAC * ng   cells (along x)
 *   ry = ACTIVE_RY_FRAC * ny   cells (along y)
 * defined by  (dx/rx)^2 + (dy/ry)^2 <= 1.
 *   - cells inside ellipse  : active
 *   - cells outside ellipse : hole / background, m = 0, frozen
 *
 * Setting  ACTIVE_RX_FRAC = ACTIVE_RY_FRAC  on a square grid recovers
 * a circle.  Setting one ≠ the other yields a stretched ellipse.
 *
 * ─── Compact active-cell execution (replaces ymsk approach) ─────────
 * Geometry is encoded as TWO compact index lists, on host and device:
 *   - d_active_ids   [n_active]   : cell indices of active cells
 *   - d_inactive_ids [n_inactive] : cell indices of hole cells
 *
 * Every per-cell kernel (RHS, Jv, build_J, apply_P) launches with one
 * thread per ACTIVE cell:
 *
 *     int tid  = blockIdx.x * blockDim.x + threadIdx.x;
 *     int cell = d_active_ids[tid];
 *     int gx   = cell % ng;
 *     int gy   = cell / ng;
 *     ... evaluate kernel at this cell ...
 *
 * No byte mask, no ymsk multiplication, no `if (active[neighbor])`
 * branching.  Hole-neighbor reads return 0 because hole cells start
 * at y=0 and stay at 0:
 *   - main() initializes y[hole] = 0
 *   - zero_inactive_kernel zeros ydot[hole] every f() call
 *   - CVODE's linear-combination updates preserve y[hole] = 0
 *
 * Inactive cells of ydot, Jv, and z are zeroed via small dedicated
 * kernels at the start of each respective wrapper.  This keeps the
 * full N_Vector consistent so SUNDIALS' inner products and norms see
 * 0 contributions from hole cells.
 *
 * ─── FFT demag still runs full-grid ─────────────────────────────────
 * Demag_Apply is full-grid by design (FFT can't be compacted).
 * For active cells the result is correct; for hole cells h_dmag[]
 * contains garbage, but the compact RHS only reads h_dmag at active
 * positions, so it doesn't matter.  Hole cells contribute 0 to the
 * input FFT (since y=0 there), so they don't contaminate the active
 * cells' output.
 *
 * ─── Build knobs ────────────────────────────────────────────────────
 *   ACTIVE_RX_FRAC : x semi-axis as fraction of ng. Default 0.25.
 *   ACTIVE_RY_FRAC : y semi-axis as fraction of ny. Default 0.25.
 *                    On a square ng×ny grid, both 0.25 → circle r=0.25*ng.
 *   DEMAG_STRENGTH : scale on FFT demag (0 disables).
 *   BLOCK_SIZE     : threads per block for compact 1D launches.
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

/* Problem constants */
#define GROUPSIZE 3

/* ─── Solver tuning knobs ─────────────────────────────────────────── */
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

/* ─── Demag controls ──────────────────────────────────────────────── */
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
#define ZERO  SUN_RCONST(0.0)

#ifndef T_TOTAL
#define T_TOTAL 1000.0
#endif

#ifndef ENABLE_OUTPUT
#define ENABLE_OUTPUT 0
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

/* ─── Initial-condition knobs (uniform) ──────────────────────────── */
#ifndef INIT_MX
#define INIT_MX 1.0
#endif
#ifndef INIT_MY
#define INIT_MY -0.0175
#endif
#ifndef INIT_MZ
#define INIT_MZ 0.0
#endif

/* ─── Output schedule ─────────────────────────────────────────────── */
#ifndef EARLY_SAVE_UNTIL
#define EARLY_SAVE_UNTIL 80.0
#endif
#ifndef EARLY_SAVE_EVERY
#define EARLY_SAVE_EVERY 5
#endif
#ifndef LATE_SAVE_EVERY
#define LATE_SAVE_EVERY 100
#endif

/* ─── Elliptical geometry knobs ──────────────────────────────────────
 * Active region: centered ellipse with semi-axes
 *     rx = ACTIVE_RX_FRAC * ng     (cells, along x)
 *     ry = ACTIVE_RY_FRAC * ny     (cells, along y)
 * Inside ellipse  (dx/rx)^2 + (dy/ry)^2 <= 1  is active.
 * Outside is masked out (hole / background).
 * ──────────────────────────────────────────────────────────────── */
#ifndef ACTIVE_RX_FRAC
#define ACTIVE_RX_FRAC 0.25
#endif
#ifndef ACTIVE_RY_FRAC
#define ACTIVE_RY_FRAC 0.25
#endif

/* ─── Material constants (device constant memory) ─────────────────── */
__constant__ sunrealtype c_msk[3] = {
    SUN_RCONST(1.0), SUN_RCONST(1.0), SUN_RCONST(1.0)};
__constant__ sunrealtype c_nsk[3] = {
    SUN_RCONST(1.0), SUN_RCONST(0.0), SUN_RCONST(0.0)};

__constant__ sunrealtype c_chk   = SUN_RCONST(1.0);
__constant__ sunrealtype c_che   = SUN_RCONST(10.0);
__constant__ sunrealtype c_alpha = SUN_RCONST(0.2);
__constant__ sunrealtype c_chg   = SUN_RCONST(1.0);

/* ─── Error checking ──────────────────────────────────────────────── */
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

/*
 * UserData
 *
 * IMPORTANT: field ORDER IS CRITICAL — jtv.cu's JtvUserData and
 * precond.cu's PcUserData mirror this layout byte-for-byte.
 *
 *   offset 0  : PrecondData *pd
 *   offset 8  : DemagData   *demag
 *   offset 16 : sunrealtype *d_hdmag           (3*ncell)
 *   offset 24 : int         *d_active_ids      (n_active)
 *   offset 32 : int         *d_inactive_ids    (n_inactive)
 *   offset 40 : int nx, ny, ng, ncell, neq     (5*4 = 20 B)
 *   offset 60 : int n_active, n_inactive       (2*4 = 8 B)
 *   offset 68 : 4 B padding before double alignment
 *   offset 72 : double nxx0, nyy0, nzz0
 *
 * Total size: 96 bytes.
 */
typedef struct {
  PrecondData  *pd;             /* offset 0  */
  DemagData    *demag;          /* offset 8  */
  sunrealtype  *d_hdmag;        /* offset 16 */
  int          *d_active_ids;   /* offset 24 */
  int          *d_inactive_ids; /* offset 32 */
  int nx;                       /* offset 40 */
  int ny;                       /* offset 44 */
  int ng;                       /* offset 48 */
  int ncell;                    /* offset 52 */
  int neq;                      /* offset 56 */
  int n_active;                 /* offset 60 */
  int n_inactive;               /* offset 64 */
  /* 4 B pad */
  double nxx0;                  /* offset 72 */
  double nyy0;                  /* offset 80 */
  double nzz0;                  /* offset 88 */
} UserData;

/* ─── SoA indexing helpers ────────────────────────────────────────── */
__host__ __device__ static inline int idx_mx(int cell, int ncell) { return cell; }
__host__ __device__ static inline int idx_my(int cell, int ncell) { return ncell + cell; }
__host__ __device__ static inline int idx_mz(int cell, int ncell) { return 2*ncell + cell; }
__host__ __device__ static inline int wrap_x(int x, int ng) {
  return (x < 0) ? (x + ng) : ((x >= ng) ? (x - ng) : x);
}
__host__ __device__ static inline int wrap_y(int y, int ny) {
  return (y < 0) ? (y + ny) : ((y >= ny) ? (y - ny) : y);
}

/*
 * zero_inactive_kernel — clears the SoA triplet (mx,my,mz) at every
 * inactive (hole) cell.  Used at the top of f(), JtvProduct, and
 * PrecondSolve to keep the hole entries of ydot / Jv / z at 0 so
 * SUNDIALS' vector ops see consistent zeros there.
 */
__global__ static void zero_inactive_kernel(
    sunrealtype* __restrict__ yd,
    const int* __restrict__ inactive_ids,
    int n_inactive,
    int ncell) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_inactive) return;
  int cell = inactive_ids[tid];
  yd[idx_mx(cell, ncell)] = ZERO;
  yd[idx_my(cell, ncell)] = ZERO;
  yd[idx_mz(cell, ncell)] = ZERO;
}

/*
 * COMPACT RHS kernel — runs over ACTIVE cells only.
 *
 * Indexing: tid → cell = active_ids[tid] → (gx, gy) = (cell%ng, cell/ng).
 *
 * Effective field:
 *     h_total = h_exchange(neighbors)
 *             + h_anisotropy(m) (per-component Landau form)
 *             + h_demag(self)   (precomputed h_dmag, full-grid FFT)
 *
 * Hole-neighbor reads return 0 automatically because y[hole] is held
 * at 0 by zero_inactive_kernel + initial conditions; no branching needed.
 */
__global__ static void f_kernel_compact(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ h_dmag,
    const int* __restrict__ active_ids,
    int n_active,
    sunrealtype* __restrict__ yd,
    int ng, int ny, int ncell) {

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_active) return;

  const int cell = active_ids[tid];
  const int gx = cell % ng;
  const int gy = cell / ng;

  const int mx = idx_mx(cell, ncell);
  const int my = idx_my(cell, ncell);
  const int mz = idx_mz(cell, ncell);

  const int xl   = wrap_x(gx - 1, ng);
  const int xr   = wrap_x(gx + 1, ng);
  const int yu   = wrap_y(gy - 1, ny);
  const int ydwn = wrap_y(gy + 1, ny);

  const int lc = gy   * ng + xl;
  const int rc = gy   * ng + xr;
  const int uc = yu   * ng + gx;
  const int dc = ydwn * ng + gx;

  const sunrealtype m1 = y[mx];
  const sunrealtype m2 = y[my];
  const sunrealtype m3 = y[mz];

  /* Neighbor reads — no branching.  Hole-cell entries are 0. */
  const sunrealtype y1L = y[idx_mx(lc, ncell)];
  const sunrealtype y1R = y[idx_mx(rc, ncell)];
  const sunrealtype y1U = y[idx_mx(uc, ncell)];
  const sunrealtype y1D = y[idx_mx(dc, ncell)];

  const sunrealtype y2L = y[idx_my(lc, ncell)];
  const sunrealtype y2R = y[idx_my(rc, ncell)];
  const sunrealtype y2U = y[idx_my(uc, ncell)];
  const sunrealtype y2D = y[idx_my(dc, ncell)];

  const sunrealtype y3L = y[idx_mz(lc, ncell)];
  const sunrealtype y3R = y[idx_mz(rc, ncell)];
  const sunrealtype y3U = y[idx_mz(uc, ncell)];
  const sunrealtype y3D = y[idx_mz(dc, ncell)];

  /* Total field (Landau anisotropy on each component, matches i5). */
  const sunrealtype h1 = c_che * (y1L + y1R + y1U + y1D)
                       + c_msk[0] * c_chk * m1 * (m1*m1 - SUN_RCONST(1.0))
                       + h_dmag[mx];
  const sunrealtype h2 = c_che * (y2L + y2R + y2U + y2D)
                       + c_msk[1] * c_chk * m2 * (m2*m2 - SUN_RCONST(1.0))
                       + h_dmag[my];
  const sunrealtype h3 = c_che * (y3L + y3R + y3U + y3D)
                       + c_msk[2] * c_chk * m3 * (m3*m3 - SUN_RCONST(1.0))
                       + h_dmag[mz];

  const sunrealtype mh = m1 * h1 + m2 * h2 + m3 * h3;

  /* No ymsk multiplication — only active cells run, so output is
   * directly the LLG RHS.  Hole entries are zeroed separately. */
  yd[mx] = c_chg * (m3 * h2 - m2 * h3) + c_alpha * (h1 - mh * m1);
  yd[my] = c_chg * (m1 * h3 - m3 * h1) + c_alpha * (h2 - mh * m2);
  yd[mz] = c_chg * (m2 * h1 - m1 * h2) + c_alpha * (h3 - mh * m3);
}

/*
 * RHS wrapper for CVODE
 *
 *   1. Demag (full-grid FFT) → h_dmag.
 *   2. zero_inactive_kernel → ydot[hole] = 0.
 *   3. compact RHS launch    → ydot[active] = LLG rhs.
 */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
  (void)t;

  UserData* udata = (UserData*)user_data;
  sunrealtype* ydata    = N_VGetDeviceArrayPointer_Cuda(y);
  sunrealtype* ydotdata = N_VGetDeviceArrayPointer_Cuda(ydot);

  if (BLOCK_SIZE > 1024) {
    fprintf(stderr, "Invalid BLOCK_SIZE=%d > 1024\n", BLOCK_SIZE);
    return -1;
  }

  /* Step 1: demag (overwrites h_dmag if active; zero-fills if off). */
  if (udata->demag && DEMAG_STRENGTH > 0.0) {
    Demag_Apply(udata->demag,
                (const double*)ydata,
                (double*)udata->d_hdmag);
  } else {
    cudaMemsetAsync(udata->d_hdmag, 0,
                    (size_t)3 * udata->ncell * sizeof(sunrealtype), 0);
  }

  /* Step 2: zero ydot at hole cells. */
  if (udata->n_inactive > 0) {
    int g0 = (udata->n_inactive + BLOCK_SIZE - 1) / BLOCK_SIZE;
    zero_inactive_kernel<<<g0, BLOCK_SIZE>>>(
        ydotdata, udata->d_inactive_ids, udata->n_inactive, udata->ncell);
  }

  /* Step 3: compact RHS at active cells. */
  if (udata->n_active > 0) {
    int g1 = (udata->n_active + BLOCK_SIZE - 1) / BLOCK_SIZE;
    f_kernel_compact<<<g1, BLOCK_SIZE>>>(
        ydata, udata->d_hdmag, udata->d_active_ids, udata->n_active,
        ydotdata, udata->ng, udata->ny, udata->ncell);
  }

  cudaError_t cuerr = cudaPeekAtLastError();
  if (cuerr != cudaSuccess) {
    fprintf(stderr, ">>> ERROR in f: kernel launch failed: %s\n",
            cudaGetErrorString(cuerr));
    return -1;
  }
  return 0;
}

/* ─── Final-stats reporter ────────────────────────────────────────── */
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
  if (t <= SUN_RCONST(EARLY_SAVE_UNTIL)) {
    return (iout % EARLY_SAVE_EVERY) == 0;
  } else {
    return (iout % LATE_SAVE_EVERY) == 0;
  }
}
#endif

/*
 * BuildEllipticalActiveLists — host-side geometry construction.
 *
 *   active   : cells inside the centered ellipse
 *              (dx/rx)^2 + (dy/ry)^2 <= 1
 *   inactive : cells outside the ellipse (frozen at m=0)
 *
 * Allocates host-side h_active_ids / h_inactive_ids; caller is
 * responsible for freeing them (via the matching free() in cleanup).
 */
static void BuildEllipticalActiveLists(int ng, int ny,
                                       int** out_h_active_ids,
                                       int** out_h_inactive_ids,
                                       int*  out_n_active,
                                       int*  out_n_inactive) {
  const int ncell = ng * ny;

  const double cx = 0.5 * (double)ng;
  const double cy = 0.5 * (double)ny;
  const double rx = ACTIVE_RX_FRAC * (double)ng;
  const double ry = ACTIVE_RY_FRAC * (double)ny;
  if (rx <= 0.0 || ry <= 0.0) {
    fprintf(stderr,
            "BuildEllipticalActiveLists: rx=%.4f ry=%.4f must be > 0\n",
            rx, ry);
    exit(EXIT_FAILURE);
  }
  const double inv_rx2 = 1.0 / (rx * rx);
  const double inv_ry2 = 1.0 / (ry * ry);

  int n_active = 0, n_inactive = 0;
  unsigned char* tmp_active =
      (unsigned char*)malloc((size_t)ncell * sizeof(unsigned char));
  if (!tmp_active) {
    fprintf(stderr, "BuildEllipticalActiveLists: malloc failed\n");
    exit(EXIT_FAILURE);
  }

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < ng; ++i) {
      const int cell = j * ng + i;
      const double dx = (double)i + 0.5 - cx;
      const double dy = (double)j + 0.5 - cy;
      const double q  = dx * dx * inv_rx2 + dy * dy * inv_ry2;

      if (q <= 1.0) {
        tmp_active[cell] = 1;
        n_active++;
      } else {
        tmp_active[cell] = 0;
        n_inactive++;
      }
    }
  }

  int* h_a = (int*)malloc((size_t)n_active * sizeof(int));
  int* h_i = (int*)malloc((size_t)n_inactive * sizeof(int));
  if ((n_active > 0 && !h_a) || (n_inactive > 0 && !h_i)) {
    fprintf(stderr, "BuildEllipticalActiveLists: id-array malloc failed\n");
    exit(EXIT_FAILURE);
  }

  int ia = 0, ii = 0;
  for (int cell = 0; cell < ncell; ++cell) {
    if (tmp_active[cell]) h_a[ia++] = cell;
    else                  h_i[ii++] = cell;
  }
  free(tmp_active);

  printf("[geometry i6] centered ellipse: rx=%.2f cells (%.4f * ng=%d), "
         "ry=%.2f cells (%.4f * ny=%d)\n",
         rx, (double)ACTIVE_RX_FRAC, ng,
         ry, (double)ACTIVE_RY_FRAC, ny);
  printf("[geometry i6] active = %d / %d (%.2f%%),  hole = %d / %d (%.2f%%)\n",
         n_active, ncell, 100.0 * n_active / ncell,
         n_inactive, ncell, 100.0 * n_inactive / ncell);

  *out_h_active_ids   = h_a;
  *out_h_inactive_ids = h_i;
  *out_n_active       = n_active;
  *out_n_inactive     = n_inactive;
}

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  SUNContext sunctx = NULL;
  sunrealtype *ydata = NULL, *abstol_data = NULL, *yhost = NULL;
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

  double dstr = 0.0, dthk = 0.0;

  int* h_active_ids   = NULL;
  int* h_inactive_ids = NULL;

  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;

  /* problem size — match i5 default */
  const int nx = 1536;
  const int ny = 512;

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
#else
  FILE* fp = NULL; (void)fp;
#endif

  udata.nx    = nx;
  udata.ny    = ny;
  udata.ng    = ng;
  udata.ncell = ncell;
  udata.neq   = neq;

  CHECK_SUNDIALS(SUNContext_Create(SUN_COMM_NULL, &sunctx));

  /* 3x3 block Jacobi preconditioner (sized for full grid; only active
   * blocks are ever read). */
  udata.pd = Precond_Create(ng, ny, ncell);
  if (!udata.pd) { fprintf(stderr, "Precond_Create failed\n"); return 1; }

  /* d_hdmag: always allocated (zero-filled when demag off). */
  CHECK_CUDA(cudaMalloc((void**)&udata.d_hdmag,
                        (size_t)3 * ncell * sizeof(sunrealtype)));
  CHECK_CUDA(cudaMemset(udata.d_hdmag, 0,
                        (size_t)3 * ncell * sizeof(sunrealtype)));

  /* ─── Build elliptical geometry: compact id lists ────────────────── */
  BuildEllipticalActiveLists(ng, ny,
                             &h_active_ids, &h_inactive_ids,
                             &udata.n_active, &udata.n_inactive);

  if (udata.n_active > 0) {
    CHECK_CUDA(cudaMalloc((void**)&udata.d_active_ids,
                          (size_t)udata.n_active * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(udata.d_active_ids, h_active_ids,
                          (size_t)udata.n_active * sizeof(int),
                          cudaMemcpyHostToDevice));
  }
  if (udata.n_inactive > 0) {
    CHECK_CUDA(cudaMalloc((void**)&udata.d_inactive_ids,
                          (size_t)udata.n_inactive * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(udata.d_inactive_ids, h_inactive_ids,
                          (size_t)udata.n_inactive * sizeof(int),
                          cudaMemcpyHostToDevice));
  }

  /* FFT demag (Newell tensor f̂, computed once here). */
  dstr = (double)DEMAG_STRENGTH;
  dthk = (double)DEMAG_THICK;
  if (dstr > 0.0) {
    udata.demag = Demag_Init(ng, ny, dthk, dstr);
    if (!udata.demag) {
      fprintf(stderr, "Demag_Init failed\n");
      Precond_Destroy(udata.pd);
      cudaFree(udata.d_hdmag);
      if (udata.d_active_ids)   cudaFree(udata.d_active_ids);
      if (udata.d_inactive_ids) cudaFree(udata.d_inactive_ids);
      free(h_active_ids);
      free(h_inactive_ids);
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
  if (y == NULL || abstol == NULL) {
    fprintf(stderr, "Failed to allocate N_Vector_Cuda objects.\n");
    goto cleanup;
  }

  FusedNVec_Init(y);

  ydata       = N_VGetHostArrayPointer_Cuda(y);
  abstol_data = N_VGetHostArrayPointer_Cuda(abstol);
  if (ydata == NULL || abstol_data == NULL) {
    fprintf(stderr, "Failed to get host array pointers from N_Vector_Cuda.\n");
    goto cleanup;
  }

  /* ─── Initial condition ───────────────────────────────────────────
   *  Active cells (inside ellipse) : m = (INIT_MX, INIT_MY, INIT_MZ),
   *                                   normalized to unit length.
   *  Hole cells (outside ellipse)  : m = 0, frozen for all time.
   *
   * Build a quick scratch byte-mask from h_active_ids so we can spot
   * active cells in O(1) during this loop.
   * ─────────────────────────────────────────────────────────────── */
  {
    unsigned char* scratch_active =
        (unsigned char*)calloc((size_t)ncell, sizeof(unsigned char));
    if (!scratch_active) {
      fprintf(stderr, "Failed to allocate scratch_active.\n");
      goto cleanup;
    }
    for (int k = 0; k < udata.n_active; ++k) {
      scratch_active[h_active_ids[k]] = 1;
    }

    double mx0 = (double)INIT_MX;
    double my0 = (double)INIT_MY;
    double mz0 = (double)INIT_MZ;
    const double n0 = sqrt(mx0*mx0 + my0*my0 + mz0*mz0);
    if (n0 > 1.0e-30) { mx0 /= n0; my0 /= n0; mz0 /= n0; }

    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < ng; i++) {
        const int cell = j * ng + i;
        const int mx_i = idx_mx(cell, ncell);
        const int my_i = idx_my(cell, ncell);
        const int mz_i = idx_mz(cell, ncell);

        if (scratch_active[cell]) {
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
    free(scratch_active);
  }

  N_VCopyToDevice_Cuda(y);
  N_VCopyToDevice_Cuda(abstol);

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

  LS = SUNLinSol_SPGMR(y, SUN_PREC_LEFT, KRYLOV_DIM, sunctx);
  if (LS == NULL) {
    fprintf(stderr, "SUNLinSol_SPGMR failed.\n");
    goto cleanup;
  }
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
  printf("Max BDF order: %d   Krylov dim: %d\n", MAX_BDF_ORDER, KRYLOV_DIM);

  printf("\n2D irregular LLG + FFT demag — CIRCULAR active geometry, "
         "compact active-cell execution\n\n");
  printf("nx=%d  ny=%d  ng=%d  ncell=%d  neq=%d\n", nx, ny, ng, ncell, neq);
  printf("BLOCK_SIZE=%d (1D compact launches)\n", BLOCK_SIZE);
  printf("periodic BC: x and y\n");
  printf("active = %d (compact),  hole = %d (frozen at m=0)\n",
         udata.n_active, udata.n_inactive);
  printf("Init: UNIFORM  m = (%.4f, %.4f, %.4f) (normalized) on active cells\n",
         (double)INIT_MX, (double)INIT_MY, (double)INIT_MZ);
  printf("DEMAG_STRENGTH=%.4f  DEMAG_THICK=%.4f  (%s)\n",
         dstr, dthk,
         dstr > 0.0 ? "h_dmag = IFFT[f_hat * m_hat] via cuFFT D2Z/Z2D, fused into RHS"
                    : "disabled (h_dmag buffer stays zero)");
  printf("T_TOTAL=%.2f  RTOL/ATOL=%.1e\n",
         (double)T_TOTAL, (double)RTOL_VAL);
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

  /* Copy final state to host and normalize active cells for output. */
  N_VCopyFromDevice_Cuda(y);
  yhost = N_VGetHostArrayPointer_Cuda(y);

  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < ng; i++) {
      int cell = j * ng + i;
      int mx_i = idx_mx(cell, ncell);
      int my_i = idx_my(cell, ncell);
      int mz_i = idx_mz(cell, ncell);

      double m1 = (double)yhost[mx_i];
      double m2 = (double)yhost[my_i];
      double m3 = (double)yhost[mz_i];
      double mag = sqrt(m1*m1 + m2*m2 + m3*m3);

      if (mag > 1e-12) {
        yhost[mx_i] = SUN_RCONST(m1 / mag);
        yhost[my_i] = SUN_RCONST(m2 / mag);
        yhost[mz_i] = SUN_RCONST(m3 / mag);
      }
    }
  }

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
  if (udata.d_hdmag)        cudaFree(udata.d_hdmag);
  if (udata.d_active_ids)   cudaFree(udata.d_active_ids);
  if (udata.d_inactive_ids) cudaFree(udata.d_inactive_ids);
  if (h_active_ids)         free(h_active_ids);
  if (h_inactive_ids)       free(h_inactive_ids);
  FusedNVec_FreePool();

#if ENABLE_OUTPUT
  if (fp) fclose(fp);
#endif

  return 0;
}
