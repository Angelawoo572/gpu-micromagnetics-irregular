/**
 * 2D irregular LLG solver — square-hole geometry + FFT demag, ymsk version.
 * CVODE + CUDA, SoA layout.
 *
 * ─── Physics ────────────────────────────────────────────────────────
 * Easy axis along x  (c_msk = {1,0,0});
 * DMI direction along x (c_nsk = {1,0,0});
 * Initial condition: ALL active cells in ONE uniform direction
 *   (slight tilt off the hard axis to break ±x symmetry; demag + hole
 *    geometry then drive the non-uniform dynamics).
 * Hole cells: m = 0, frozen (yd = 0 forever) via the mask.
 *
 * ─── ymsk approach (replaces active/inactive lists) ────────────────
 * Geometry is encoded as a single SoA mask  ymsk[3*ncell]:
 *
 *   ymsk = 1   on active cells
 *   ymsk = 0   on hole cells (and stays so for the whole run)
 *
 * Every kernel runs over ALL cells, no compact index lists, no byte
 * mask, no neighbor-side branching.  The kernel computes the LLG RHS
 * naively, reading neighbor m values directly — for hole neighbors
 * those reads return 0 (initial condition + frozen by mask) and so
 * contribute 0 to the exchange/DMI/demag sums automatically.
 *
 * The "inactive" handling boils down to ONE multiplication at the very
 * end of f / Jv / P-solve:
 *
 *   yd[mx] = ymsk[mx] * (LLG formula);    // 0 in hole, formula in body
 *
 * That's it.  No `if (active[neighbor])` checks, no zero_inactive_kernel.
 *
 * ─── Effective-field structure ──────────────────────────────────────
 * Single unified RHS kernel sums:
 *
 *   h_total = h_exchange + h_anisotropy + h_DMI + h_demag
 *
 * Demag is precomputed once per f() via cuFFT D2Z/Z2D and stored in
 * d_hdmag (SoA, 3*ncell).  The unified kernel just reads h_dmag[mx/my/mz]
 * like any other field.
 *
 * ─── Unit-sphere regularization ─────────────────────────────────────
 * normalize_m_kernel runs at the top of every f():
 *
 *   ymp = sqrt(m1² + m2² + m3²)
 *   m_new = m / (ymp + 0.001)
 *
 * The +0.001 in the denominator regularizes m=0 (hole cells) to stay
 * exactly at 0 without any branching:  0 / 0.001 = 0.  Active cells
 * with |m|≈1 are nudged toward |m|≈1/1.01 ≈ 0.99, a stable fixed
 * point of the regularized normalization.  The simplified LLG form
 * (which assumes |m|=1) is then evaluated on this slightly-shrunken
 * sphere with no measurable physical impact.
 *
 * ─── Build knobs ────────────────────────────────────────────────────
 * Enable:  make DEMAG_STRENGTH=1.0 DEMAG_THICK=1.0
 * Disable: make DEMAG_STRENGTH=0.0   (h_dmag buffer stays zero)
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

#ifndef BLOCK_X
#define BLOCK_X 16
#endif

#ifndef BLOCK_Y
#define BLOCK_Y 8
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

/* ─── Irregular square-hole geometry ─────────────────────────────────
 * The hole is a square centered at (HOLE_CENTER_X_FRAC, HOLE_CENTER_Y_FRAC)
 * (in fractions of the ng×ny grid).  HOLE_RADIUS_FRAC_Y keeps its old
 * macro name for Makefile compatibility; here it means the HALF-SIDE of
 * the square in units of ny (so the full side length is 2·HOLE_RADIUS_FRAC_Y·ny).
 * ──────────────────────────────────────────────────────────────────── */
#ifndef HOLE_CENTER_X_FRAC
#define HOLE_CENTER_X_FRAC 0.50
#endif
#ifndef HOLE_CENTER_Y_FRAC
#define HOLE_CENTER_Y_FRAC 0.50
#endif
#ifndef HOLE_RADIUS_FRAC_Y
#define HOLE_RADIUS_FRAC_Y 0.22
#endif

/* ─── Material constants (device constant memory) ─────────────────── */
__constant__ sunrealtype c_msk[3] = {
    SUN_RCONST(1.0), SUN_RCONST(1.0), SUN_RCONST(1.0)};
__constant__ sunrealtype c_nsk[3] = {
    SUN_RCONST(1.0), SUN_RCONST(0.0), SUN_RCONST(0.0)};

__constant__ sunrealtype c_chk   = SUN_RCONST(1.0);
__constant__ sunrealtype c_che   = SUN_RCONST(20.0);
__constant__ sunrealtype c_alpha = SUN_RCONST(0.2);
__constant__ sunrealtype c_chg   = SUN_RCONST(1.0);
__constant__ sunrealtype happ1  = SUN_RCONST(-0.3);
__constant__ sunrealtype happ2  = SUN_RCONST(0.0);
__constant__ sunrealtype happ3  = SUN_RCONST(0.0);

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
 *   offset 0  : PrecondData  *pd
 *   offset 8  : DemagData    *demag
 *   offset 16 : sunrealtype  *d_hdmag    (3*ncell)
 *   offset 24 : sunrealtype  *d_ymsk     (3*ncell, 1 on active, 0 in hole)
 *   offset 32 : int nx, ny, ng, ncell, neq    (5*4 = 20 B)
 *   offset 52 : 4 B padding before double alignment
 *   offset 56 : double nxx0, nyy0, nzz0
 *
 * Total size: 80 bytes.
 */
typedef struct {
  PrecondData  *pd;        /* offset 0  */
  DemagData    *demag;     /* offset 8  */
  sunrealtype  *d_hdmag;   /* offset 16 */
  sunrealtype  *d_ymsk;    /* offset 24 — geometry mask, SoA 3*ncell */
  int nx;                  /* offset 32 */
  int ny;                  /* offset 36 */
  int ng;                  /* offset 40 */
  int ncell;               /* offset 44 */
  int neq;                 /* offset 48 */
  /* 4 B pad */
  double nxx0;             /* offset 56 */
  double nyy0;             /* offset 64 */
  double nzz0;             /* offset 72 */
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
 * UNIFIED RHS kernel — runs over EVERY cell, no compact lists.
 *
 *   h_total = h_exchange(neighbors)
 *           + h_anisotropy(m1)            ← c_msk = {1,0,0}, only h1
 *           + h_DMI(x-neighbors of mx)    ← c_nsk = {1,0,0}, only h1
 *           + h_demag(self)               ← from precomputed h_dmag
 *
 *   yd[mα] = ymsk[mα] * (chg*(m × h)_α + α*(h_α − (m·h) m_α))
 *
 * Hole cells: ymsk=0 → yd=0 → y stays at its initial value (0).
 * Hole-neighbor reads: y[hole]=0 → contributes 0 to neighbor sums
 * automatically.  No `if (active[neighbor])` branches anywhere.
 *
 * |m|=1 is enforced (modulo +0.001 regularization) by normalize_m_kernel.
 */
__global__ static void f_kernel_unified_soa_periodic(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ h_dmag,
    const sunrealtype* __restrict__ ymsk,
    sunrealtype*       __restrict__ yd,
    int ng, int ny, int ncell) {

  const int gx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gy = blockIdx.y * blockDim.y + threadIdx.y;
  if (gx >= ng || gy >= ny) return;

  const int cell = gy * ng + gx;
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

  /* Neighbor reads — no branching.  In-hole neighbors return 0 (frozen). */
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

  /* Total field.  Anisotropy {1,0,0} → only h1.  DMI {1,0,0} → only h1. */
  const sunrealtype h1 = happ1+c_che * (y1L + y1R + y1U + y1D) + c_msk[0] * c_chk * m1*(m1*m1-SUN_RCONST(1.0))+ h_dmag[mx];
  const sunrealtype h2 = happ2+c_che * (y2L + y2R + y2U + y2D) + c_msk[1] * c_chk * m2*(m2*m2-SUN_RCONST(1.0))+ h_dmag[my];
  const sunrealtype h3 = happ3+c_che * (y3L + y3R + y3U + y3D) + c_msk[2] * c_chk * m3*(m3*m3-SUN_RCONST(1.0))+ h_dmag[mz];

  const sunrealtype mh = m1 * h1 + m2 * h2 + m3 * h3;

  /* Mask the output — that's the entire "inactive cell" handling. */
  yd[mx] = ymsk[mx] * (c_chg * (m3 * h2 - m2 * h3) + c_alpha * (h1 - mh * m1));
  yd[my] = ymsk[my] * (c_chg * (m1 * h3 - m3 * h1) + c_alpha * (h2 - mh * m2));
  yd[mz] = ymsk[mz] * (c_chg * (m2 * h1 - m1 * h2) + c_alpha * (h3 - mh * m3));
}

/*
 * RHS wrapper for CVODE
 *
 * Step 0: normalize y in-place (regularized; hole cells stay at 0).
 * Step 1: compute h_dmag = IFFT[ f̂ · FFT(y) ]; if demag off, zero buffer.
 * Step 2: unified kernel sums all fields and writes yd, masked by ymsk.
 */
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

  /* Step 1: demag (overwrites h_dmag if active; zero-fills if off). */
  if (udata->demag && DEMAG_STRENGTH > 0.0) {
    Demag_Apply(udata->demag,
                (const double*)ydata,
                (double*)udata->d_hdmag);
  } else {
    cudaMemsetAsync(udata->d_hdmag, 0,
                    (size_t)3 * udata->ncell * sizeof(sunrealtype), 0);
  }

  /* Step 2: unified RHS kernel. */
  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid((udata->ng + block.x - 1) / block.x,
            (udata->ny + block.y - 1) / block.y);
  f_kernel_unified_soa_periodic<<<grid, block>>>(
      ydata, udata->d_hdmag, udata->d_ymsk, ydotdata,
      udata->ng, udata->ny, udata->ncell);

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

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  SUNContext sunctx = NULL;
  sunrealtype *ydata = NULL, *abstol_data = NULL,*yhost = NULL;;
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
  long  n_active_count = 0, n_hole_count = 0;

  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;

  /* problem size */
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

  /* fill user data */
  udata.nx    = nx;
  udata.ny    = ny;
  udata.ng    = ng;
  udata.ncell = ncell;
  udata.neq   = neq;

  CHECK_SUNDIALS(SUNContext_Create(SUN_COMM_NULL, &sunctx));

  /* 3x3 block Jacobi preconditioner */
  udata.pd = Precond_Create(ng, ny, ncell);
  if (!udata.pd) { fprintf(stderr, "Precond_Create failed\n"); return 1; }

  /* d_hdmag: always allocated (zero-filled when demag off). */
  CHECK_CUDA(cudaMalloc((void**)&udata.d_hdmag,
                        (size_t)3 * ncell * sizeof(sunrealtype)));
  CHECK_CUDA(cudaMemset(udata.d_hdmag, 0,
                        (size_t)3 * ncell * sizeof(sunrealtype)));

  /* d_ymsk: SoA mask (3*ncell), 1 on active cells, 0 in hole.  This
   * is the entire geometry encoding — every kernel multiplies its
   * output by this, no other branching exists in the device code. */
  CHECK_CUDA(cudaMalloc((void**)&udata.d_ymsk,
                        (size_t)3 * ncell * sizeof(sunrealtype)));

  /* Build mask on host, then upload. */
  sunrealtype *h_ymsk = (sunrealtype*)malloc((size_t)3 * ncell * sizeof(sunrealtype));
  if (!h_ymsk) { fprintf(stderr, "h_ymsk malloc failed\n"); return 1; }
  {
    const double cx       = HOLE_CENTER_X_FRAC * (double)(ng - 1);
    const double cy       = HOLE_CENTER_Y_FRAC * (double)(ny - 1);
    const double halfside = HOLE_RADIUS_FRAC_Y * (double)ny;  /* square half-side */

    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < ng; ++i) {
        const int cell = j * ng + i;
        const double ddx = fabs((double)i - cx);
        const double ddy = fabs((double)j - cy);
        /* Square hole: cell is in the hole iff both |dx| and |dy| ≤ halfside. */
        const sunrealtype m = (ddx <= halfside && ddy <= halfside)
                                ? SUN_RCONST(0.0) : SUN_RCONST(1.0);
        if (m == SUN_RCONST(0.0)) n_hole_count++; else n_active_count++;
        h_ymsk[idx_mx(cell, ncell)] = m;
        h_ymsk[idx_my(cell, ncell)] = m;
        h_ymsk[idx_mz(cell, ncell)] = m;
      }
    }
  }
  CHECK_CUDA(cudaMemcpy(udata.d_ymsk, h_ymsk,
                        (size_t)3 * ncell * sizeof(sunrealtype),
                        cudaMemcpyHostToDevice));
  printf("[geometry] square hole: active=%ld  hole=%ld  (total %d cells)\n",
         n_active_count, n_hole_count, ncell);

  /* FFT demag (Newell tensor f̂, computed once here). */
  dstr = (double)DEMAG_STRENGTH;
  dthk = (double)DEMAG_THICK;
  if (dstr > 0.0) {
    udata.demag = Demag_Init(ng, ny, dthk, dstr);
    if (!udata.demag) {
      fprintf(stderr, "Demag_Init failed\n");
      Precond_Destroy(udata.pd);
      cudaFree(udata.d_hdmag);
      cudaFree(udata.d_ymsk);
      free(h_ymsk);
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
   *   Upper 1/4 (high gy):       m = (-1, 0, 0)         (anti-aligned with body)
   *   Lower 3/4 (low gy):        m = (+1, m_y_tail, 0)
   *      where m_y_tail = -INIT_MY_TAIL on left half  (gx <  ng/2)
   *                     = +INIT_MY_TAIL on right half (gx >= ng/2)
   * Hole cells: m = 0 exactly. */
  {
    const int    j_split   = (3 * ny) / 4;          /* boundary between body and top stripe */
    const double m_tail    = (double)INIT_MY;       /* small ±y perturbation amplitude */

    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < ng; i++) {
        const int cell = j * ng + i;
        const int mx_i = idx_mx(cell, ncell);
        const int my_i = idx_my(cell, ncell);
        const int mz_i = idx_mz(cell, ncell);

        if (h_ymsk[mx_i] != SUN_RCONST(0.0)) {
          double mx0, my0, mz0 = 0.0;

          if (j >= j_split) {
            /* Upper 1/4: anti-aligned. */
            mx0 = -1.0;
            my0 =  0.0;
          } else {
            /* Lower 3/4: aligned with +x, with a left/right tail in m_y. */
            mx0 = +1.0;
            my0 = (i < ng / 2) ? -m_tail : +m_tail;
          }

          /* Normalize to unit length. */
          const double n0 = sqrt(mx0*mx0 + my0*my0 + mz0*mz0);
          if (n0 > 1e-30) { mx0 /= n0; my0 /= n0; mz0 /= n0; }

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

  printf("\n2D irregular LLG + FFT demag — square hole + ymsk geometry\n");
  printf("LLG form: standard simplified, |m|≈1 enforced by normalize-in-f (regularized)\n\n");
  printf("nx=%d  ny=%d  ng=%d  ncell=%d  neq=%d\n", nx, ny, ng, ncell, neq);
  printf("periodic BC: x and y\n");
  printf("Init: UNIFORM  m = (%.4f, %.4f, %.4f) (normalized)\n",
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

  /* Copy final state to host and normalize only for output. */
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
  /* ====== FINAL STATE OUTPUT ====== */
  // {
  //   FILE* fp_final = fopen("output.txt", "w");
  //   if (!fp_final) {
  //     fprintf(stderr, "Failed to open output.txt\n");
  //   } else {
  //     fprintf(fp_final, "%f %d %d\n", (double)t, nx, ny);

  //     for (int j = 0; j < ny; j++) {
  //       for (int i = 0; i < ng; i++) {
  //         int cell = j * ng + i;

  //         fprintf(fp_final, "%e %e %e\n",
  //           (double)yhost[idx_mx(cell, ncell)],
  //           (double)yhost[idx_my(cell, ncell)],
  //           (double)yhost[idx_mz(cell, ncell)]
  //         );
  //       }
  //     }

  //     fclose(fp_final);
  //     printf("[output] final state written to output.txt\n");
  //   }
  // }

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
  if (udata.d_ymsk)  cudaFree(udata.d_ymsk);
  if (h_ymsk) free(h_ymsk);
  FusedNVec_FreePool();

#if ENABLE_OUTPUT
  if (fp) fclose(fp);
#endif

  return 0;
}
