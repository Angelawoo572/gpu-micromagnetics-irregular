/**
 * 2D periodic LLG solver — head-on transition initial condition
 * CVODE + CUDA, SoA layout
 *
 * ─── Physics ────────────────────────────────────────────────────────
 * Easy axis along x (c_msk = {1,0,0}); DMI direction along x (c_nsk = {1,0,0}).
 * Initial condition: three-stripe head-on transition along x
 *   i ∈ [0,       ng/4):  mx = -1
 *   i ∈ [ng/4,   3ng/4):  mx = +1
 *   i ∈ [3ng/4,   ng  ):  mx = -1
 * Small random (my, mz) to break symmetry.
 *
 * ─── Effective-field structure ──────────────────────────────────────
 * Every RHS cell sees ONE total field assembled inside a single kernel:
 *
 *   h_total = h_exchange + h_anisotropy + h_DMI + h_demag
 *
 * Demag is NOT a post-processing step. It is precomputed per f() as
 *
 *   h_dmag = IFFT[ f̂(k) · ŷ(k) ]        ← f̂ constant, computed once
 *
 * and then read inside the unified kernel just like any other SoA field.
 *
 * ─── Unit-sphere constraint (normalize-in-f) ────────────────────────
 * The standard LLG form  dm/dt = γ(m×h) + α(h − (m·h)m)  assumes
 * |m|=1.  BDF interpolation can drift |m| away from 1, and if m·h < 0
 * (common in domain walls with strong demag) the drift is amplified
 * exponentially, causing CVODE to stall.
 *
 * Solution (per professor): at the top of every f() call, normalize y
 * in-place so |m|=1 at every cell:
 *
 *   ymp = sqrt(y[mx]² + y[my]² + y[mz]²)
 *   y[mx] /= ymp;  y[my] /= ymp;  y[mz] /= ymp
 *
 * This is a lightweight GPU kernel (~2 µs) that projects onto the unit
 * sphere before the RHS kernel sees y.  The simplified LLG form is then
 * always evaluated exactly on the manifold — no modified formulas needed.
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

/* ─── Initial-condition knobs (head-on three-stripe) ──────────────── */
#ifndef STRIPE_LEFT_FRAC
#define STRIPE_LEFT_FRAC 0.25     /* first wall at ng * STRIPE_LEFT_FRAC  */
#endif

#ifndef STRIPE_RIGHT_FRAC
#define STRIPE_RIGHT_FRAC 0.75    /* second wall at ng * STRIPE_RIGHT_FRAC */
#endif

#ifndef INIT_RANDOM_EPS
#define INIT_RANDOM_EPS 0.01      /* amplitude of my/mz perturbation     */
#endif

#ifndef INIT_RANDOM_SEED
#define INIT_RANDOM_SEED 12345    /* reproducibility                     */
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

#ifndef NX_VAL
#define NX_VAL 1536
#endif

#ifndef NY_VAL
#define NY_VAL 512
#endif

/* ─── Material constants (device constant memory) ─────────────────── */
/* Easy axis along x: only c_msk[0] is non-zero, and anisotropy feeds off m1.
 * DMI direction (c_nsk) stays along x. */
__constant__ sunrealtype c_msk[3] = {
    SUN_RCONST(1.0), SUN_RCONST(0.0), SUN_RCONST(0.0)};
__constant__ sunrealtype c_nsk[3] = {
    SUN_RCONST(1.0), SUN_RCONST(0.0), SUN_RCONST(0.0)};

__constant__ sunrealtype c_chk   = SUN_RCONST(4.0);
__constant__ sunrealtype c_che   = SUN_RCONST(4.0);
__constant__ sunrealtype c_alpha = SUN_RCONST(0.2);
__constant__ sunrealtype c_chg   = SUN_RCONST(1.0);
__constant__ sunrealtype c_cha   = SUN_RCONST(0.0);
__constant__ sunrealtype c_chb   = SUN_RCONST(0.3);

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
 * IMPORTANT: field ORDER IS CRITICAL.
 *   - PrecondData *pd       at offset 0  (PrecondSetup/Solve cast rule)
 *   - DemagData   *demag    at offset 8
 *   - sunrealtype *d_hdmag  at offset 16 — always allocated, zero if demag off
 *   - ints follow at offset 24+
 *   - doubles (self-coupling Nαα(0)*strength) at the tail
 *
 * jtv.cu's JtvUserData and precond.cu's PcUserData mirror this layout
 * byte-for-byte. When adding/reordering fields, update both mirrors.
 */
typedef struct {
  PrecondData  *pd;        /* offset 0  — must be first */
  DemagData    *demag;     /* offset 8  */
  sunrealtype  *d_hdmag;   /* offset 16 — device buffer, 3*ncell */
  int nx;                  /* offset 24 */
  int ny;                  /* offset 28 */
  int ng;                  /* offset 32 */
  int ncell;               /* offset 36 */
  int neq;                 /* offset 40 */
  /* 4 bytes padding before double alignment */
  double nxx0;             /* offset 48 — Nxx(0) * demag_strength */
  double nyy0;             /* offset 56 — Nyy(0) * demag_strength */
  double nzz0;             /* offset 64 — Nzz(0) * demag_strength */
} UserData;

/* ─── SoA indexing helpers ────────────────────────────────────────── */
__host__ __device__ static inline int idx_mx(int cell, int ncell) {
  return cell;
}
__host__ __device__ static inline int idx_my(int cell, int ncell) {
  return ncell + cell;
}
__host__ __device__ static inline int idx_mz(int cell, int ncell) {
  return 2 * ncell + cell;
}
__host__ __device__ static inline int wrap_x(int x, int ng) {
  return (x < 0) ? (x + ng) : ((x >= ng) ? (x - ng) : x);
}
__host__ __device__ static inline int wrap_y(int y, int ny) {
  return (y < 0) ? (y + ny) : ((y >= ny) ? (y - ny) : y);
}

/*
 * normalize_m_kernel
 *
 * Per-cell in-place normalization of y so that |m| = 1 exactly:
 *
 *   ymp = sqrt(y[mx]² + y[my]² + y[mz]²)
 *   y[mx] /= ymp
 *   y[my] /= ymp
 *   y[mz] /= ymp
 *
 * Called at the top of every f() evaluation, BEFORE the RHS kernel.
 * This ensures the simplified LLG form (which assumes |m|=1) is always
 * evaluated on the unit sphere, regardless of BDF interpolation drift.
 *
 * Cost: one lightweight kernel launch (~2 µs at ncell=65536) per f() call.
 * The normalize is applied to the CVODE-owned y vector in-place; CVODE
 * passes y as non-const to f(), so this is legal.  The BDF interpolant
 * itself is not modified — only the evaluation point seen by f().
 */
__global__ static void normalize_m_kernel(
    sunrealtype* __restrict__ y,
    int ncell) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= ncell) return;

  const int mx = idx_mx(cell, ncell);
  const int my = idx_my(cell, ncell);
  const int mz = idx_mz(cell, ncell);

  const sunrealtype m1 = y[mx];
  const sunrealtype m2 = y[my];
  const sunrealtype m3 = y[mz];

  const sunrealtype ymp = sqrt(m1 * m1 + m2 * m2 + m3 * m3);

  if (ymp > SUN_RCONST(1.0e-30)) {
    const sunrealtype inv_ymp = SUN_RCONST(1.0) / ymp;
    y[mx] = m1 * inv_ymp;
    y[my] = m2 * inv_ymp;
    y[mz] = m3 * inv_ymp;
  }
}

/*
 * UNIFIED RHS kernel
 *
 * Reads y (self + 4 neighbors) and h_dmag (self only) from global memory,
 * assembles the TOTAL effective field in one pass:
 *
 *   h_total = h_exchange(neighbors)
 *           + h_anisotropy(m1)
 *           + h_DMI(x-neighbors)
 *           + h_demag(self)         ← precomputed via FFT outside
 *
 * then writes the standard LLG update for this cell:
 *
 *   dm/dt = γ (m × h) + α ( h − (m·h) m )
 *
 * This is the original simplified form that assumes |m|=1.  The assumption
 * is enforced by normalize_m_kernel which runs at the top of every f() call,
 * projecting y back to the unit sphere before the RHS is evaluated.
 */
__global__ static void f_kernel_unified_soa_periodic(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ h_dmag,
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

  const int left_cell  = gy   * ng + xl;
  const int right_cell = gy   * ng + xr;
  const int up_cell    = yu   * ng + gx;
  const int down_cell  = ydwn * ng + gx;

  const sunrealtype m1 = y[mx];
  const sunrealtype m2 = y[my];
  const sunrealtype m3 = y[mz];

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

  /* Assemble TOTAL field — demag is just another SoA field in the sum.
   * Anisotropy couples to m1 (easy axis along x): c_msk = {1,0,0}. */
  const sunrealtype h1 =
      c_che * (y[lx] + y[rx] + y[ux] + y[dx]) +   /* exchange    */
      c_msk[0] * (c_chk * m1 + c_cha) +           /* anisotropy  */
      c_chb * c_nsk[0] * (y[lx] + y[rx]) +        /* DMI         */
      h_dmag[mx];                                 /* demag (FFT) */

  const sunrealtype h2 =
      c_che * (y[ly] + y[ry] + y[uy] + y[dy]) +
      c_msk[1] * (c_chk * m1 + c_cha) +
      c_chb * c_nsk[1] * (y[ly] + y[ry]) +
      h_dmag[my];

  const sunrealtype h3 =
      c_che * (y[lz] + y[rz] + y[uz] + y[dz]) +
      c_msk[2] * (c_chk * m1 + c_cha) +
      c_chb * c_nsk[2] * (y[lz] + y[rz]) +
      h_dmag[mz];

  /* Standard LLG (|m|=1 enforced by normalize_m_kernel before this). */
  const sunrealtype mh = m1 * h1 + m2 * h2 + m3 * h3;

  yd[mx] = c_chg * (m3 * h2 - m2 * h3) + c_alpha * (h1 - mh * m1);
  yd[my] = c_chg * (m1 * h3 - m3 * h1) + c_alpha * (h2 - mh * m2);
  yd[mz] = c_chg * (m2 * h1 - m1 * h2) + c_alpha * (h3 - mh * m3);
}

/*
 * RHS wrapper for CVODE
 *
 * Step 1: compute h_dmag = IFFT[ f̂ · FFT(y) ]
 *         f̂ is precomputed once in Demag_Init, never recomputed.
 *         If demag is disabled, h_dmag stays zero → contributes nothing.
 *
 * Step 2: single unified kernel sums all fields (exchange + anisotropy
 *         + DMI + demag) and writes ydot.
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

  /* Step 0: normalize y in-place so |m| = 1 at every cell.
   * BDF interpolation can drift |m| away from 1; this projection
   * ensures the simplified LLG (which assumes |m|=1) is evaluated
   * on the unit sphere.  Cost: ~2 µs per call at ncell=65536. */
  {
    const int nb = 256; // 8 warps
    const int ng_norm = (udata->ncell + nb - 1) / nb;
    normalize_m_kernel<<<ng_norm, nb>>>(ydata, udata->ncell);
  }

  /* Step 1: demag field (constant f̂ × current m̂, inverse FFT).
   *
   * When demag is active, Demag_Apply's unshift_h_kernel OVERWRITES
   * h_out (not +=), so the memset is redundant — skip it.
   * When demag is off, zero the buffer once so the unified kernel
   * reads zeros from h_dmag[mx/my/mz] and demag contributes nothing. */
  if (udata->demag && DEMAG_STRENGTH > 0.0) {
    Demag_Apply(udata->demag,
                (const double*)ydata,
                (double*)udata->d_hdmag);
  } else {
    cudaMemsetAsync(udata->d_hdmag, 0,
                    (size_t)3 * udata->ncell * sizeof(sunrealtype), 0);
  }

  /* Step 2: unified kernel — one pass over y + h_dmag → ydot. */
  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid((udata->ng + block.x - 1) / block.x,
            (udata->ny + block.y - 1) / block.y);

  f_kernel_unified_soa_periodic<<<grid, block>>>(
      ydata, udata->d_hdmag, ydotdata,
      udata->ng, udata->ny, udata->ncell);

  cudaError_t cuerr = cudaPeekAtLastError();
  if (cuerr != cudaSuccess) {
    fprintf(stderr, ">>> ERROR in f: kernel launch failed: %s\n",
            cudaGetErrorString(cuerr));
    return -1;
  }

  return 0;
}

/* ─── Final-stats reporter (unchanged) ────────────────────────────── */
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

  /* problem size */
  const int nx = NX_VAL;
  const int ny = NY_VAL;

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

  /* ─── Stripe geometry for head-on initial condition ─── */
  const int q1 = (int)(STRIPE_LEFT_FRAC  * (double)ng + 0.5);
  const int q3 = (int)(STRIPE_RIGHT_FRAC * (double)ng + 0.5);

  /* fill user data */
  udata.nx    = nx;
  udata.ny    = ny;
  udata.ng    = ng;
  udata.ncell = ncell;
  udata.neq   = neq;
  udata.pd      = NULL;
  udata.demag   = NULL;
  udata.d_hdmag = NULL;
  udata.nxx0    = 0.0;
  udata.nyy0    = 0.0;
  udata.nzz0    = 0.0;

  CHECK_SUNDIALS(SUNContext_Create(SUN_COMM_NULL, &sunctx));

  /* 3x3 block Jacobi preconditioner */
  udata.pd = Precond_Create(ng, ny, ncell);
  if (!udata.pd) { fprintf(stderr, "Precond_Create failed\n"); return 1; }

  /* ─── Always allocate d_hdmag ──────────────────────────────────────
   * The unified kernel always reads h_dmag[mx/my/mz]. If demag is
   * disabled, the buffer is simply zeroed every f() call and contributes
   * nothing to h_total. This is simpler than having two kernel paths.
   * ─────────────────────────────────────────────────────────────────── */
  CHECK_CUDA(cudaMalloc((void**)&udata.d_hdmag,
                        (size_t)3 * ncell * sizeof(sunrealtype)));
  CHECK_CUDA(cudaMemset(udata.d_hdmag, 0,
                        (size_t)3 * ncell * sizeof(sunrealtype)));

  /* FFT demag (Newell tensor f̂, computed once here). */
  const double dstr = (double)DEMAG_STRENGTH;
  const double dthk = (double)DEMAG_THICK;
  if (dstr > 0.0) {
    /* Demag_Init takes (nx, ny) where nx is number of columns. Here
     * ng = nx/GROUPSIZE is the number of physical columns in the grid. */
    udata.demag = Demag_Init(ng, ny, dthk, dstr);
    if (!udata.demag) {
      fprintf(stderr, "Demag_Init failed\n");
      Precond_Destroy(udata.pd);
      cudaFree(udata.d_hdmag);
      return 1;
    }
    /* Pull N(0)·strength into UserData so the preconditioner can include
     * the demag self-coupling in its local 3x3 Jacobian block. Off-diagonals
     * of N at r=0 vanish by 4-fold symmetry of the cell face integral. */
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

  /* ─── Initial condition: head-on three-stripe along x ──────────────
   *
   *   i ∈ [0,  q1) : mx = -1      ← pointing -x
   *   i ∈ [q1, q3) : mx = +1      ← pointing +x
   *   i ∈ [q3, ng) : mx = -1      ← pointing -x
   *
   *   q1 = STRIPE_LEFT_FRAC  * ng   (default 0.25 * ng)
   *   q3 = STRIPE_RIGHT_FRAC * ng   (default 0.75 * ng)
   *
   * my, mz get small uniform random values of amplitude INIT_RANDOM_EPS
   * to break the y-z symmetry; after normalization |m| = 1. Under LLG
   * relaxation these perturbations are what allow the chevron/zigzag
   * domain-wall structure (Fig. 5.7) to emerge from the sharp step.
   * ─────────────────────────────────────────────────────────────────── */
  {
    const double eps = (double)INIT_RANDOM_EPS;
    srand((unsigned int)INIT_RANDOM_SEED);

    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < ng; i++) {
        cell = j * ng + i;

        const int mx_i = idx_mx(cell, ncell);
        const int my_i = idx_my(cell, ncell);
        const int mz_i = idx_mz(cell, ncell);

        /* Dominant mx by x-position. */
        double mx0 = (i >= q1 && i < q3) ? 1.0 : -1.0;

        /* Uniform random perturbation in [-eps, +eps] for my, mz. */
        double my0 = eps * (2.0 * (double)rand() / (double)RAND_MAX - 1.0);
        double mz0 = eps * (2.0 * (double)rand() / (double)RAND_MAX - 1.0);

        /* Normalize so |m| = 1. */
        const double norm = sqrt(mx0 * mx0 + my0 * my0 + mz0 * mz0);
        mx0 /= norm; my0 /= norm; mz0 /= norm;

        ydata[mx_i] = SUN_RCONST(mx0);
        ydata[my_i] = SUN_RCONST(my0);
        ydata[mz_i] = SUN_RCONST(mz0);

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

  printf("\n2D periodic LLG — head-on transition + unified-field RHS\n");
  printf("LLG form: standard (damping = alpha*(h - (m.h)*m)), |m|=1 enforced by normalize-in-f\n\n");
  printf("nx=%d  ny=%d  ng=%d  ncell=%d  neq=%d\n", nx, ny, ng, ncell, neq);
  printf("periodic BC: x and y\n");
  printf("Init: head-on stripes   q1=%d (x=%.2f*ng)   q3=%d (x=%.2f*ng)\n",
         q1, (double)STRIPE_LEFT_FRAC, q3, (double)STRIPE_RIGHT_FRAC);
  printf("      mx = -1 | +1 | -1       random my,mz amplitude = %.3f\n",
         (double)INIT_RANDOM_EPS);
  printf("DEMAG_STRENGTH=%.4f  DEMAG_THICK=%.4f  (%s)\n",
         dstr, dthk,
         dstr > 0.0 ? "h_dmag = IFFT[f_hat * m_hat] via cuFFT real-to-complex (D2Z/Z2D), fused into RHS"
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
  FusedNVec_FreePool();

#if ENABLE_OUTPUT
  if (fp) fclose(fp);
#endif

  return 0;
}
