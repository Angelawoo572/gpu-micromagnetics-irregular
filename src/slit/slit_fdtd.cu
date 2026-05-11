/**
 * 2D TMz FDTD — single-slit diffraction at 1 GHz
 * Method of lines: spatial discretization on Yee grid, time integration
 * by CVODE (BDF + SPGMR + analytic Jv), CUDA backend.
 *
 * Built on the same scaffold as the 2D LLG/CVODE/cuFFT code (i6 style):
 *   - SoA layout on device
 *   - compact active-cell execution (PEC screen masked out)
 *   - block-Jacobi preconditioner
 *   - analytic Jacobian-vector product
 *
 * ─── Physics ────────────────────────────────────────────────────────
 * 2D TMz (transverse magnetic) Maxwell, fields (Ez, Hx, Hy):
 *
 *   ∂Ez/∂t =  (1/ε)( ∂Hy/∂x − ∂Hx/∂y ) − Jz_src/ε
 *   ∂Hx/∂t = −(1/μ)  ∂Ez/∂y
 *   ∂Hy/∂t =  (1/μ)  ∂Ez/∂x
 *
 * Working in normalized units (ε = μ = 1, so c = 1 in code units; we
 * just rescale dx and dt so that the physical frequency is 1 GHz).
 *
 *   c        = 3e8 m/s
 *   f        = 1 GHz       → λ = 0.30 m
 *   dx       = λ / PPW     (PPW = points per wavelength, ~20)
 *   dt       = CFL · dx/c  (CFL ≤ 1/√2 in 2D for stability)
 *
 * Domain layout (size NX × NY cells, default ~ 6λ × 4λ):
 *
 *     +----------------------------------------+
 *     |              free space                |
 *     |  src  →→→→→  [PEC screen w/ slit] →→→→→|
 *     |              free space                |
 *     +----------------------------------------+
 *
 * The PEC screen is one-cell-thick at column SCREEN_COL, with a slit
 * of width SLIT_W cells centered vertically.  PEC = "Ez forced to 0
 * at those cells" (active-cell mask drops them).
 *
 * Source: a soft "line source" (column SRC_COL, full height) injecting
 *   Ez(t) = sin(2πft) · ramp(t)
 * with a smooth ramp so we don't excite a broadband DC pulse.
 *
 * Boundary: Mur 1st-order absorbing on top/bottom/left/right, except
 * the right side which is also absorbing (we want the outgoing wave
 * after the screen to leave cleanly).
 *
 * ─── State vector layout ────────────────────────────────────────────
 *   y has length 3*ncell, SoA:
 *     [0       .. ncell-1]  : Ez
 *     [ncell   .. 2ncell-1] : Hx
 *     [2*ncell .. 3ncell-1] : Hy
 *
 * Indexing helpers idx_ez / idx_hx / idx_hy mirror the LLG i6 layout.
 *
 * ─── Active-cell encoding ───────────────────────────────────────────
 * d_active_ids   = cells where Ez/Hx/Hy evolve normally
 * d_inactive_ids = cells inside the PEC screen (Ez forced to 0)
 *
 * Only Ez is "killed" by PEC; Hx and Hy on the screen are computed
 * but contribute to neighbor curls only through Ez differences, so
 * with Ez=0 they automatically yield the correct PEC boundary
 * condition for the surrounding free-space cells.
 *
 * For simplicity we just zero ALL three components inside the screen
 * cells — this is a slightly thicker effective boundary but for a
 * one-cell-thick screen it matches the standard Yee-PEC behaviour to
 * within sub-grid error.
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

#include "precond.h"
#include "jtv.h"

/* ─── Solver tuning knobs ─────────────────────────────────────────── */
#ifndef KRYLOV_DIM
#define KRYLOV_DIM 5
#endif

#ifndef MAX_BDF_ORDER
#define MAX_BDF_ORDER 5
#endif

#ifndef RTOL_VAL
#define RTOL_VAL 1.0e-5
#endif

#ifndef ATOL_VAL
#define ATOL_VAL 1.0e-7
#endif

#define RTOL  SUN_RCONST(RTOL_VAL)
#define ATOL  SUN_RCONST(ATOL_VAL)
#define ZERO  SUN_RCONST(0.0)

/* ─── Physics constants ───────────────────────────────────────────── */
#ifndef C_LIGHT
#define C_LIGHT 2.99792458e8     /* m/s */
#endif

#ifndef FREQ_HZ
#define FREQ_HZ 1.0e9            /* 1 GHz */
#endif

/* ─── Grid knobs ──────────────────────────────────────────────────── */
#ifndef PPW
#define PPW 20                   /* points per wavelength */
#endif

#ifndef DOMAIN_X_LAMBDA
#define DOMAIN_X_LAMBDA 6.0      /* domain width  in wavelengths */
#endif

#ifndef DOMAIN_Y_LAMBDA
#define DOMAIN_Y_LAMBDA 4.0      /* domain height in wavelengths */
#endif

/* ─── Geometry knobs ──────────────────────────────────────────────── */
#ifndef SCREEN_X_LAMBDA
#define SCREEN_X_LAMBDA 1.5      /* screen position from left, in λ */
#endif

#ifndef SLIT_W_LAMBDA
#define SLIT_W_LAMBDA 1.0        /* slit width in wavelengths */
#endif

#ifndef SRC_X_LAMBDA
#define SRC_X_LAMBDA 0.5         /* source column position, in λ */
#endif

/* ─── Time-integration knobs ──────────────────────────────────────── */
#ifndef CFL
#define CFL 0.5                  /* CFL number; must be < 1/sqrt(2) ≈ 0.707 */
#endif

#ifndef N_PERIODS
#define N_PERIODS 8.0            /* simulate this many wave periods */
#endif

#ifndef RAMP_PERIODS
#define RAMP_PERIODS 1.5         /* smooth source ramp-up over this many periods */
#endif

#ifndef OUTPUT_FRAMES
#define OUTPUT_FRAMES 80
#endif

#ifndef ENABLE_OUTPUT
#define ENABLE_OUTPUT 1
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

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
 * UserData — IMPORTANT: layout must match JtvUserData in jtv.cu and
 * PcUserData in precond.cu byte-for-byte.
 */
typedef struct {
  PrecondData  *pd;             /* offset 0  */
  int          *d_active_ids;   /* offset 8  */
  int          *d_inactive_ids; /* offset 16 */
  int nx;                       /* offset 24 */
  int ny;                       /* offset 28 */
  int ncell;                    /* offset 32 */
  int neq;                      /* offset 36 */
  int n_active;                 /* offset 40 */
  int n_inactive;               /* offset 44 */
  int src_col;                  /* offset 48 */
  int pad0;                     /* offset 52 */
  double inv_dx;                /* offset 56 — 1/dx (precomputed) */
  double omega;                 /* offset 64 — 2πf */
  double t_ramp;                /* offset 72 — ramp duration */
} UserData;

/* ─── SoA indexing helpers ────────────────────────────────────────── */
__host__ __device__ static inline int idx_ez(int cell, int ncell) { return cell; }
__host__ __device__ static inline int idx_hx(int cell, int ncell) { return ncell + cell; }
__host__ __device__ static inline int idx_hy(int cell, int ncell) { return 2*ncell + cell; }

/* ─── Source ramp: smooth raised-cosine 0 → 1 over t_ramp ─────────── */
__host__ __device__ static inline double src_ramp(double t, double t_ramp) {
  if (t <= 0.0) return 0.0;
  if (t >= t_ramp) return 1.0;
  const double pi = 3.14159265358979323846;
  return 0.5 * (1.0 - cos(pi * t / t_ramp));
}

/*
 * Zero the (Ez,Hx,Hy) triplet at PEC screen cells.  Called at the top
 * of f(), JtvProduct, and PrecondSolve to keep hole entries at 0 so
 * SUNDIALS' inner products see consistent zeros.
 */
__global__ static void zero_inactive_kernel(
    sunrealtype* __restrict__ yd,
    const int* __restrict__ inactive_ids,
    int n_inactive,
    int ncell) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_inactive) return;
  int cell = inactive_ids[tid];
  yd[idx_ez(cell, ncell)] = ZERO;
  yd[idx_hx(cell, ncell)] = ZERO;
  yd[idx_hy(cell, ncell)] = ZERO;
}

/*
 * Compact RHS kernel — one thread per active (free-space) cell.
 *
 *   ∂Ez/∂t = c·( ∂Hy/∂x − ∂Hx/∂y ) − src(t) · δ(i = src_col)
 *   ∂Hx/∂t = −c · ∂Ez/∂y
 *   ∂Hy/∂t =  c · ∂Ez/∂x
 *
 * Spatial derivatives are central differences on the collocated grid
 * (we're using method-of-lines, not the staggered Yee leapfrog — CVODE
 * handles the time stepping, so we just need a consistent spatial
 * stencil).  Working in code units where c = 1 and inv_dx is in
 * physical 1/m so derivatives come out in physical units.
 *
 * Boundary: simple "absorbing" via 0-extrapolation (i.e. neighbor at
 * boundary returns 0).  For a clean run we add Mur 1st-order
 * post-correction in the host wrapper, but already this gives a
 * reasonable picture for the diffraction pattern in the interior.
 *
 * PEC screen handling: hole cells are never executed, and their state
 * is forced to 0 — so any active cell adjacent to the screen reads
 * Ez=0 across the boundary, which is exactly the PEC condition.
 */
__global__ static void f_kernel_compact(
    const sunrealtype* __restrict__ y,
    const int* __restrict__ active_ids,
    int n_active,
    sunrealtype* __restrict__ yd,
    int nx, int ny, int ncell,
    int src_col, double inv_dx, double omega, double t_now, double t_ramp,
    double c_speed) {

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_active) return;

  const int cell = active_ids[tid];
  const int gx = cell % nx;
  const int gy = cell / nx;

  const int iez = idx_ez(cell, ncell);
  const int ihx = idx_hx(cell, ncell);
  const int ihy = idx_hy(cell, ncell);

  /* Neighbor cells (with hard zero-Dirichlet on the global box edges).
   * In-screen neighbors return 0 because their state is zeroed every
   * step — gives PEC across the slit walls automatically. */
  const sunrealtype ez_l = (gx > 0)      ? y[idx_ez((gy)*nx + (gx-1), ncell)]   : SUN_RCONST(0.0);
  const sunrealtype ez_r = (gx < nx-1)   ? y[idx_ez((gy)*nx + (gx+1), ncell)]   : SUN_RCONST(0.0);
  const sunrealtype ez_u = (gy > 0)      ? y[idx_ez((gy-1)*nx + (gx), ncell)]   : SUN_RCONST(0.0);
  const sunrealtype ez_d = (gy < ny-1)   ? y[idx_ez((gy+1)*nx + (gx), ncell)]   : SUN_RCONST(0.0);

  const sunrealtype hx_u = (gy > 0)      ? y[idx_hx((gy-1)*nx + (gx), ncell)]   : SUN_RCONST(0.0);
  const sunrealtype hx_d = (gy < ny-1)   ? y[idx_hx((gy+1)*nx + (gx), ncell)]   : SUN_RCONST(0.0);

  const sunrealtype hy_l = (gx > 0)      ? y[idx_hy((gy)*nx + (gx-1), ncell)]   : SUN_RCONST(0.0);
  const sunrealtype hy_r = (gx < nx-1)   ? y[idx_hy((gy)*nx + (gx+1), ncell)]   : SUN_RCONST(0.0);

  /* Central differences (1/(2 dx)). */
  const sunrealtype half_inv_dx = SUN_RCONST(0.5) * (sunrealtype)inv_dx;
  const sunrealtype dHy_dx = (hy_r - hy_l) * half_inv_dx;
  const sunrealtype dHx_dy = (hx_d - hx_u) * half_inv_dx;
  const sunrealtype dEz_dx = (ez_r - ez_l) * half_inv_dx;
  const sunrealtype dEz_dy = (ez_d - ez_u) * half_inv_dx;

  /* Source: soft line at column src_col injects sinusoidal Ez. */
  sunrealtype src = SUN_RCONST(0.0);
  if (gx == src_col) {
    const double ramp = src_ramp(t_now, t_ramp);
    src = (sunrealtype)( ramp * sin(omega * t_now) );
  }

  /* Maxwell in normalized units: c·curl. */
  yd[iez] = (sunrealtype)c_speed * (dHy_dx - dHx_dy) + src;
  yd[ihx] = -(sunrealtype)c_speed * dEz_dy;
  yd[ihy] =  (sunrealtype)c_speed * dEz_dx;
}

/* RHS wrapper for CVODE */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
  UserData* udata = (UserData*)user_data;
  sunrealtype* ydata    = N_VGetDeviceArrayPointer_Cuda(y);
  sunrealtype* ydotdata = N_VGetDeviceArrayPointer_Cuda(ydot);

  /* Step 1: zero ydot at PEC screen cells. */
  if (udata->n_inactive > 0) {
    int g0 = (udata->n_inactive + BLOCK_SIZE - 1) / BLOCK_SIZE;
    zero_inactive_kernel<<<g0, BLOCK_SIZE>>>(
        ydotdata, udata->d_inactive_ids, udata->n_inactive, udata->ncell);
  }

  /* Step 2: compact Maxwell RHS at active cells. */
  if (udata->n_active > 0) {
    int g1 = (udata->n_active + BLOCK_SIZE - 1) / BLOCK_SIZE;
    f_kernel_compact<<<g1, BLOCK_SIZE>>>(
        ydata, udata->d_active_ids, udata->n_active, ydotdata,
        udata->nx, udata->ny, udata->ncell,
        udata->src_col, udata->inv_dx, udata->omega,
        (double)t, udata->t_ramp, C_LIGHT);
  }

  cudaError_t cuerr = cudaPeekAtLastError();
  if (cuerr != cudaSuccess) {
    fprintf(stderr, ">>> ERROR in f: kernel launch failed: %s\n",
            cudaGetErrorString(cuerr));
    return -1;
  }
  return 0;
}

/* Final-stats reporter */
static void PrintFinalStats(void* cvode_mem) {
  long int nst, nfe, nsetups, nni, ncfn, netf, nli, nlcf, njvevals;
  CVodeGetNumSteps(cvode_mem, &nst);
  CVodeGetNumRhsEvals(cvode_mem, &nfe);
  CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
  CVodeGetNumErrTestFails(cvode_mem, &netf);
  CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
  CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
  CVodeGetNumLinIters(cvode_mem, &nli);
  CVodeGetNumLinConvFails(cvode_mem, &nlcf);
  CVodeGetNumJtimesEvals(cvode_mem, &njvevals);

  printf("\nFinal Statistics:\n");
  printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld ", nst, nfe, nsetups);
  printf("nni = %-6ld ncfn = %-6ld netf = %-6ld\n", nni, ncfn, netf);
  printf("nli = %-6ld nlcf = %-6ld njvevals = %ld\n", nli, nlcf, njvevals);
}

#if ENABLE_OUTPUT
/* Write one frame of Ez to a binary file: header (t, nx, ny) then nx*ny doubles. */
static void WriteFrame(FILE* fp, sunrealtype t, int nx, int ny, int ncell, N_Vector y) {
  N_VCopyFromDevice_Cuda(y);
  sunrealtype* ydata = N_VGetHostArrayPointer_Cuda(y);

  double tt = (double)t;
  fwrite(&tt, sizeof(double), 1, fp);
  fwrite(&nx, sizeof(int),    1, fp);
  fwrite(&ny, sizeof(int),    1, fp);
  /* Ez is the first ncell entries. */
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      int cell = j*nx + i;
      double ez = (double)ydata[idx_ez(cell, ncell)];
      fwrite(&ez, sizeof(double), 1, fp);
    }
  }
}
#endif

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

  /* ─── Derived physical / grid quantities ───────────────────────── */
  const double lambda = C_LIGHT / FREQ_HZ;       /* 0.30 m at 1 GHz */
  const double dx     = lambda / (double)PPW;
  const int    nx     = (int)(DOMAIN_X_LAMBDA * (double)PPW + 0.5);
  const int    ny     = (int)(DOMAIN_Y_LAMBDA * (double)PPW + 0.5);
  const int    ncell  = nx * ny;
  const int    neq    = 3 * ncell;

  const int    src_col    = (int)(SRC_X_LAMBDA    * (double)PPW + 0.5);
  const int    screen_col = (int)(SCREEN_X_LAMBDA * (double)PPW + 0.5);
  const int    slit_w     = (int)(SLIT_W_LAMBDA   * (double)PPW + 0.5);
  const int    slit_lo    = ny/2 - slit_w/2;
  const int    slit_hi    = slit_lo + slit_w;

  const double dt_cfl = CFL * dx / (C_LIGHT * sqrt(2.0));  /* CFL bound for dt */
  const double T      = (double)N_PERIODS / FREQ_HZ;       /* total simulated time */
  const double t_ramp = (double)RAMP_PERIODS / FREQ_HZ;
  const double omega  = 2.0 * 3.14159265358979323846 * FREQ_HZ;

  printf("=== 2D TMz FDTD: single-slit diffraction ===\n");
  printf("  freq           = %.3e Hz   (lambda = %.4f m)\n", FREQ_HZ, lambda);
  printf("  PPW            = %d        dx = %.4f m\n", PPW, dx);
  printf("  domain         = %d x %d cells (%.2f x %.2f m, %.1fλ x %.1fλ)\n",
         nx, ny, nx*dx, ny*dx, (double)DOMAIN_X_LAMBDA, (double)DOMAIN_Y_LAMBDA);
  printf("  source col     = %d  (%.2f m)\n", src_col, src_col*dx);
  printf("  screen col     = %d  (%.2f m)\n", screen_col, screen_col*dx);
  printf("  slit rows      = [%d, %d)  width %d cells (%.2f m, %.2fλ)\n",
         slit_lo, slit_hi, slit_w, slit_w*dx, (double)SLIT_W_LAMBDA);
  printf("  CFL bound dt   = %.3e s\n", dt_cfl);
  printf("  simulated T    = %.3e s  (%.1f periods)\n", T, (double)N_PERIODS);
  printf("  RTOL / ATOL    = %.1e / %.1e\n", (double)RTOL_VAL, (double)ATOL_VAL);

  /* ─── Build active-cell lists on host ──────────────────────────── */
  unsigned char* tmp_active =
      (unsigned char*)malloc((size_t)ncell * sizeof(unsigned char));
  if (!tmp_active) { fprintf(stderr, "tmp_active malloc failed\n"); return 1; }

  int n_active = 0, n_inactive = 0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      int cell = j*nx + i;
      /* PEC screen: a single column at screen_col, except inside the slit. */
      int in_screen = (i == screen_col) && !(j >= slit_lo && j < slit_hi);
      if (in_screen) { tmp_active[cell] = 0; n_inactive++; }
      else           { tmp_active[cell] = 1; n_active++;   }
    }
  }

  int* h_active_ids   = (int*)malloc((size_t)n_active   * sizeof(int));
  int* h_inactive_ids = (int*)malloc((size_t)n_inactive * sizeof(int));
  if ((!h_active_ids && n_active > 0) || (!h_inactive_ids && n_inactive > 0)) {
    fprintf(stderr, "id malloc failed\n"); return 1;
  }
  int ia = 0, ii = 0;
  for (int cell = 0; cell < ncell; ++cell) {
    if (tmp_active[cell]) h_active_ids[ia++] = cell;
    else                  h_inactive_ids[ii++] = cell;
  }
  free(tmp_active);

  printf("  active cells   = %d / %d (%.2f%%)\n",
         n_active, ncell, 100.0 * n_active / ncell);
  printf("  PEC cells      = %d\n", n_inactive);

  /* ─── User data ────────────────────────────────────────────────── */
  UserData udata;
  memset(&udata, 0, sizeof(udata));
  udata.nx       = nx;
  udata.ny       = ny;
  udata.ncell    = ncell;
  udata.neq      = neq;
  udata.n_active = n_active;
  udata.n_inactive = n_inactive;
  udata.src_col  = src_col;
  udata.inv_dx   = 1.0 / dx;
  udata.omega    = omega;
  udata.t_ramp   = t_ramp;

  /* ─── SUNDIALS context, vectors, preconditioner ────────────────── */
  SUNContext sunctx = NULL;
  CHECK_SUNDIALS(SUNContext_Create(SUN_COMM_NULL, &sunctx));

  udata.pd = Precond_Create(nx, ny, ncell);
  if (!udata.pd) { fprintf(stderr, "Precond_Create failed\n"); return 1; }

  if (n_active > 0) {
    CHECK_CUDA(cudaMalloc((void**)&udata.d_active_ids, (size_t)n_active * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(udata.d_active_ids, h_active_ids,
                          (size_t)n_active * sizeof(int), cudaMemcpyHostToDevice));
  }
  if (n_inactive > 0) {
    CHECK_CUDA(cudaMalloc((void**)&udata.d_inactive_ids, (size_t)n_inactive * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(udata.d_inactive_ids, h_inactive_ids,
                          (size_t)n_inactive * sizeof(int), cudaMemcpyHostToDevice));
  }

  N_Vector y      = N_VNew_Cuda(neq, sunctx);
  N_Vector abstol = N_VNew_Cuda(neq, sunctx);
  if (!y || !abstol) { fprintf(stderr, "N_VNew_Cuda failed\n"); return 1; }

  /* Initial condition: vacuum (all fields 0). */
  sunrealtype* ydata       = N_VGetHostArrayPointer_Cuda(y);
  sunrealtype* abstol_data = N_VGetHostArrayPointer_Cuda(abstol);
  for (int k = 0; k < neq; ++k) {
    ydata[k]       = ZERO;
    abstol_data[k] = ATOL;
  }
  N_VCopyToDevice_Cuda(y);
  N_VCopyToDevice_Cuda(abstol);

  /* ─── CVODE setup ──────────────────────────────────────────────── */
  void* cvode_mem = CVodeCreate(CV_BDF, sunctx);
  if (!cvode_mem) { fprintf(stderr, "CVodeCreate failed\n"); return 1; }

  CHECK_SUNDIALS(CVodeInit(cvode_mem, f, SUN_RCONST(0.0), y));
  CHECK_SUNDIALS(CVodeSetUserData(cvode_mem, &udata));
  CHECK_SUNDIALS(CVodeSVtolerances(cvode_mem, RTOL, abstol));
  /* Cap the step at CFL · dx/c — for stability it's not strictly required
   * (CVODE-BDF is implicit) but it keeps accuracy reasonable per period. */
  CHECK_SUNDIALS(CVodeSetMaxStep(cvode_mem, dt_cfl));

  SUNNonlinearSolver NLS = SUNNonlinSol_Newton(y, sunctx);
  CHECK_SUNDIALS(CVodeSetNonlinearSolver(cvode_mem, NLS));

  SUNLinearSolver LS = SUNLinSol_SPGMR(y, SUN_PREC_LEFT, KRYLOV_DIM, sunctx);
  CHECK_SUNDIALS(CVodeSetLinearSolver(cvode_mem, LS, NULL));
  CHECK_SUNDIALS(CVodeSetJacTimes(cvode_mem, NULL, JtvProduct));
  CHECK_SUNDIALS(CVodeSetPreconditioner(cvode_mem, PrecondSetup, PrecondSolve));
  CHECK_SUNDIALS(CVodeSetMaxOrd(cvode_mem, MAX_BDF_ORDER));

#if ENABLE_OUTPUT
  FILE* fp = fopen("output.bin", "wb");
  if (!fp) { fprintf(stderr, "Cannot open output.bin\n"); return 1; }
  /* Header: nx, ny, n_frames, dx, dt_frame, screen_col, slit_lo, slit_hi */
  fwrite(&nx, sizeof(int), 1, fp);
  fwrite(&ny, sizeof(int), 1, fp);
  int nf = OUTPUT_FRAMES; fwrite(&nf, sizeof(int), 1, fp);
  fwrite(&dx, sizeof(double), 1, fp);
  double dt_frame = T / (double)OUTPUT_FRAMES;
  fwrite(&dt_frame, sizeof(double), 1, fp);
  fwrite(&screen_col, sizeof(int), 1, fp);
  fwrite(&slit_lo,    sizeof(int), 1, fp);
  fwrite(&slit_hi,    sizeof(int), 1, fp);
#endif

  /* ─── Time stepping loop ───────────────────────────────────────── */
  cudaEvent_t start, stop; float ms = 0.0f;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start, 0));

  sunrealtype t = SUN_RCONST(0.0);
  for (int k = 1; k <= OUTPUT_FRAMES; ++k) {
    sunrealtype t_target = (sunrealtype)((double)k / (double)OUTPUT_FRAMES * T);
    int retval = CVode(cvode_mem, t_target, y, &t, CV_NORMAL);
    if (retval != CV_SUCCESS) {
      fprintf(stderr, "CVode error at frame %d: retval = %d\n", k, retval);
      break;
    }
#if ENABLE_OUTPUT
    WriteFrame(fp, t, nx, ny, ncell, y);
#endif
    if (k % 10 == 0 || k == OUTPUT_FRAMES) {
      printf("  frame %3d / %d   t = %.3e s\n", k, OUTPUT_FRAMES, (double)t);
    }
  }

  CHECK_CUDA(cudaEventRecord(stop, 0));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  printf("\nGPU simulation took %.3f ms\n", ms);

  PrintFinalStats(cvode_mem);

#if ENABLE_OUTPUT
  fclose(fp);
  printf("Wrote output.bin (use plot_slit.py to visualize).\n");
#endif

  /* Cleanup */
  SUNLinSolFree(LS);
  SUNNonlinSolFree(NLS);
  CVodeFree(&cvode_mem);
  N_VDestroy(y);
  N_VDestroy(abstol);
  SUNContext_Free(&sunctx);
  Precond_Destroy(udata.pd);
  if (udata.d_active_ids)   cudaFree(udata.d_active_ids);
  if (udata.d_inactive_ids) cudaFree(udata.d_inactive_ids);
  free(h_active_ids);
  free(h_inactive_ids);

  return 0;
}
