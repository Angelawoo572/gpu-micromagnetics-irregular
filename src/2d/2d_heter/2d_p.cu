/**
 * 2D periodic smooth-texture LLG solver
 * CVODE + CUDA, SoA layout
 *
 * Tier 1 + Tier 3 optimisation applied via fused_nvec.h / fused_nvec.cu:
 *   - N_VEnableFusedOps_Cuda  (Tier 1, SUNDIALS built-in fused ops)
 *   - FusedNVec_Init          (Tier 3, custom multi-dot / linear-comb /
 *                              scale-add-multi kernels + propagating clone)
 *
 * Changes from the baseline are marked  // <<< FUSED
 *
 * Geometry / topology:
 *   - full regular 2D grid
 *   - periodic in x
 *   - periodic in y
 *   - no masked / inactive cells
 *
 * Initialization:
 *   - one circular smooth texture in the interior
 *   - center approximately points downward (mz < 0)
 *   - far field approximately points upward (mz > 0)
 *   - transition region rotates continuously
 *   - in-plane component is radial (Néel-like)
 *
 * Output policy:
 *   - write initial frame at t = 0
 *   - dense output during early transient
 *   - sparse output after that
 *
 * SoA layout in one N_Vector:
 *   [mx for all cells][my for all cells][mz for all cells]
 */

#include <cvode/cvode.h>
#include <cuda_runtime.h>
#include <math.h>
#include <nvector/nvector_cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>
#include <sundials/sundials_iterative.h>   /* SUN_CLASSICAL_GS */

#include "deferred_nvector.h"   /* <<< FUSED: Tier 1 + Tier 3 header */
#include "precond.h"        /* <<< PRECOND: 3x3 block Jacobi preconditioner */
#include "jtv.h"           /* <<< JTV:    analytic Jacobian-times-vector */

/* Problem constants */
#define GROUPSIZE 3

/* -------------------------------------------------------
 * Solver tuning knobs (set via Makefile or -D flags)
 *
 * KRYLOV_DIM   : max Krylov dimension for SPGMR.
 *                0 = SUNDIALS default (min(neq, 5)).
 *                With a good preconditioner, 3 often suffices.
 *
 * MAX_BDF_ORDER: max BDF order.  5 = CVODE default.
 *                2 reduces per-step vector-op count at cost of
 *                possibly more steps (test both for your problem).
 * ------------------------------------------------------- */
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

/* Circle parameters */
#ifndef CIRCLE_CENTER_X_FRAC
#define CIRCLE_CENTER_X_FRAC 0.50
#endif

#ifndef CIRCLE_CENTER_Y_FRAC
#define CIRCLE_CENTER_Y_FRAC 0.50
#endif

/* radius measured in cell units, relative to ny */
#ifndef CIRCLE_RADIUS_FRAC_Y
#define CIRCLE_RADIUS_FRAC_Y 0.22
#endif

/* smooth texture initialization parameters */
#ifndef TEXTURE_CORE_MZ
#define TEXTURE_CORE_MZ -0.998
#endif

#ifndef TEXTURE_OUTER_MZ
#define TEXTURE_OUTER_MZ 0.998
#endif

/* transition width as a fraction of radius */
#ifndef TEXTURE_WIDTH_FRAC
#define TEXTURE_WIDTH_FRAC 0.35
#endif

/* tiny bias to avoid exact singularity at center */
#ifndef TEXTURE_EPS
#define TEXTURE_EPS 1.0e-12
#endif

/* output schedule:
 * write more densely before EARLY_SAVE_UNTIL,
 * then more sparsely afterward.
 */
#ifndef EARLY_SAVE_UNTIL
#define EARLY_SAVE_UNTIL 80.0
#endif

#ifndef EARLY_SAVE_EVERY
#define EARLY_SAVE_EVERY 5
#endif

#ifndef LATE_SAVE_EVERY
#define LATE_SAVE_EVERY 100
#endif

/* typed constant memory */
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

/* error checking */
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

/* user data
 *
 * IMPORTANT: PrecondData *pd MUST be the first field.
 * PrecondSetup/PrecondSolve in precond.cu cast user_data to
 * PrecondData** and dereference it to get pd. This only works
 * correctly when pd is at offset 0 in the struct.
 */
typedef struct {
  PrecondData *pd;  /* first field: 3x3 block Jacobi preconditioner data   */
  int nx;           /* old scalar width = 3 * ng                           */
  int ny;           /* number of rows                                       */
  int ng;           /* number of physical cells per row                    */
  int ncell;        /* total physical cells = ng * ny                      */
  int neq;          /* total equations = 3 * ncell                         */
} UserData;

/* SoA indexing helpers */
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
 * RHS kernel — unchanged from baseline
 *
 * Mapping:
 *   one thread -> one physical cell
 *
 * Boundary policy:
 *   periodic in x and y (toroidal domain)
 */
__global__ static void f_kernel_group_soa_periodic(
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

  const int xl   = wrap_x(gx - 1, ng);
  const int xr   = wrap_x(gx + 1, ng);
  const int yu   = wrap_y(gy - 1, ny);
  const int ydwn = wrap_y(gy + 1, ny);

  const int left_cell  = gy * ng + xl;
  const int right_cell = gy * ng + xr;
  const int up_cell    = yu * ng + gx;
  const int down_cell  = ydwn * ng + gx;

  /* local m */
  const sunrealtype m1 = y[mx];
  const sunrealtype m2 = y[my];
  const sunrealtype m3 = y[mz];

  /* neighbor component indices */
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

  const sunrealtype h1 =
      c_che * (y[lx] + y[rx] + y[ux] + y[dx]) +
      c_msk[0] * (c_chk * m3 + c_cha) +
      c_chb * c_nsk[0] * (y[lx] + y[rx]);

  const sunrealtype h2 =
      c_che * (y[ly] + y[ry] + y[uy] + y[dy]) +
      c_msk[1] * (c_chk * m3 + c_cha) +
      c_chb * c_nsk[1] * (y[ly] + y[ry]);

  const sunrealtype h3 =
      c_che * (y[lz] + y[rz] + y[uz] + y[dz]) +
      c_msk[2] * (c_chk * m3 + c_cha) +
      c_chb * c_nsk[2] * (y[lz] + y[rz]);

  const sunrealtype mh = m1 * h1 + m2 * h2 + m3 * h3;

  yd[mx] = c_chg * (m3 * h2 - m2 * h3) + c_alpha * (h1 - mh * m1);
  yd[my] = c_chg * (m1 * h3 - m3 * h1) + c_alpha * (h2 - mh * m2);
  yd[mz] = c_chg * (m2 * h1 - m1 * h2) + c_alpha * (h3 - mh * m3);
}

/* RHS wrapper for CVODE — unchanged */
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

  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid((udata->ng + block.x - 1) / block.x,
            (udata->ny + block.y - 1) / block.y);

  f_kernel_group_soa_periodic<<<grid, block>>>(
      ydata, ydotdata, udata->ng, udata->ny, udata->ncell);

  cudaError_t cuerr = cudaPeekAtLastError();
  if (cuerr != cudaSuccess) {
    fprintf(stderr, ">>> ERROR in f: kernel launch failed: %s\n",
            cudaGetErrorString(cuerr));
    return -1;
  }

  return 0;
}

/*
 * Preconditioner callbacks are defined in precond.cu / precond.h.
 * They implement a 3x3 block-diagonal Jacobi preconditioner:
 *   P_cell = I - gamma * J_local  (analytic 3x3 block per cell)
 *   psetup: compute P^{-1} via Cramer's rule, store 9 doubles/cell
 *   psolve: z = P^{-1} r  (one matrix-vector multiply per cell)
 *
 * Why 3x3 block beats scalar Jacobi
 * ------------------------------------
 * Scalar version used M_i = 1 + gamma*alpha*|H|^2 (same for mx,my,mz).
 * It ignores the precession cross-terms:
 *   J[0][1] = -c_chg*h3  (dominant for large grids)
 *   J[1][0] = +c_chg*h3
 *   J[0][2], J[1][2], J[2][0], J[2][1]  (damping cross-terms)
 * Without these, GMRES still needs 4-5 iterations.
 * With 3x3 block: 1-2 iterations suffice.
 * Expected speedup for large problems: 2-3x over scalar precond.
 */

/* final stats */
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
  CVodeGetNumJtimesEvals(cvode_mem, &njvevals);   /* <<< JTV: analytic Jv calls */

  printf("\nFinal Statistics:\n");
  printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld ", nst, nfe, nsetups);
  printf("nni = %-6ld ncfn = %-6ld netf = %-6ld nge = %ld\n",
         nni, ncfn, netf, nge);
  printf("nli = %-6ld nlcf = %-6ld njvevals = %ld  "
         "(analytic Jv; FD would have cost %ld extra f() calls)\n",
         nli, nlcf, njvevals, njvevals);
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

  int cell;

  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;

  /* problem size */
  const int nx = 3000;
  const int ny = 1280;

  if (nx % GROUPSIZE != 0) {
    fprintf(stderr, "nx must be a multiple of GROUPSIZE=%d\n", GROUPSIZE);
    return 1;
  }

  const int ng    = nx / GROUPSIZE;
  const int ncell = ng * ny;
  const int neq   = 3 * ncell;

  FILE* fp = NULL;
#if ENABLE_OUTPUT
  fp = fopen("output.txt", "w");
  if (fp == NULL) {
    fprintf(stderr, "Error opening output file.\n");
    return 1;
  }
  setvbuf(fp, NULL, _IOFBF, 1 << 20);
#endif

  /* circle geometry */
  const double cx = CIRCLE_CENTER_X_FRAC * (double)(ng - 1);
  const double cy = CIRCLE_CENTER_Y_FRAC * (double)(ny - 1);
  const double radius = CIRCLE_RADIUS_FRAC_Y * (double)ny;

  /* fill user data */
  udata.nx    = nx;
  udata.ny    = ny;
  udata.ng    = ng;
  udata.ncell = ncell;
  udata.neq   = neq;
  udata.pd = NULL;

  CHECK_SUNDIALS(SUNContext_Create(SUN_COMM_NULL, &sunctx));

  /* allocate 3x3 block Jacobi preconditioner (9 doubles per cell) */
  udata.pd = Precond_Create(ng, ny, ncell);   /* <<< PRECOND: 3x3 block */
  if (!udata.pd) { fprintf(stderr, "Precond_Create failed\n"); return 1; }

  y      = N_VNew_Cuda(neq, sunctx);
  abstol = N_VNew_Cuda(neq, sunctx);
  if (y == NULL || abstol == NULL) {
    fprintf(stderr, "Failed to allocate N_Vector_Cuda objects.\n");
#if ENABLE_OUTPUT
    fclose(fp);
#endif
    return 1;
  }

  /* FUSED  Tier 1 + Tier 3: install fused ops on y.
   *
   * Must be called AFTER N_VNew_Cuda and BEFORE CVodeInit,
   * so that all vectors CVODE clones from y (Krylov basis,
   * work arrays, weight/error vectors) inherit the overrides.
   * We do NOT call it on abstol — abstol is read-only for
   * tolerances and is not cloned for arithmetic work.
   */
  FusedNVec_Init(y);   /* FUSED */

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

  /* Initialize y and abstol in SoA layout
   * Smooth Néel-like radial texture:
   *   - center: approximately downward
   *   - far away: approximately upward
   *   - transition region: continuous rotation
   */
  {
    const double core_mz  = (double)TEXTURE_CORE_MZ;
    const double outer_mz = (double)TEXTURE_OUTER_MZ;

    double width = (double)TEXTURE_WIDTH_FRAC * radius;
    if (width < 1.0) width = 1.0;

    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < ng; i++) {
        cell = j * ng + i;

        const int mx = idx_mx(cell, ncell);
        const int my = idx_my(cell, ncell);
        const int mz = idx_mz(cell, ncell);

        const double dx = (double)i - cx;
        const double dy = (double)j - cy;
        const double rho = sqrt(dx * dx + dy * dy);

        const double u = (rho - radius) / width;
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

  /* Linear solver: SPGMR with LEFT Jacobi preconditioner
   *
   * SUN_PREC_LEFT: CVODE applies M^{-1} to the residual before
   * each GMRES iteration.  PrecondSetup builds the per-cell 3x3
   * block P=(I-gamma*J_local) and inverts it (Cramer's rule).
   * PrecondSolve applies z = P^{-1} r (one matmul per cell).
   *
   * Expected effect on large problems (bandwidth-limited):
   *   Without precond: SPGMR needs K~5 Krylov iters to converge.
   *   With Jacobi:     K~1-2 iters suffice.
   *   - linearSumKernel, dotProdKernel, wL2NormSquare calls drop ~60%.
   *
   * KRYLOV_DIM=0 → SUNDIALS default (min(neq, SUNSPGMR_MAXL_DEFAULT=5))
   * Set KRYLOV_DIM=3 in Makefile for a tighter cap.
   */
  LS = SUNLinSol_SPGMR(y, SUN_PREC_LEFT, KRYLOV_DIM, sunctx);
  if (LS == NULL) {
    fprintf(stderr, "SUNLinSol_SPGMR failed.\n");
    goto cleanup;
  }
  CHECK_SUNDIALS(CVodeSetLinearSolver(cvode_mem, LS, NULL));
  /* JTV: register analytic Jv — eliminates ~20K extra f() calls.
   * First arg NULL means no jtsetup callback needed.
   * user_data (UserData*) is already registered via CVodeSetUserData;
   * JtvProduct casts it to JtvUserData* which has the same memory layout
   * for the fields it uses (ng, ny, ncell). */
  CHECK_SUNDIALS(CVodeSetJacTimes(cvode_mem, NULL, JtvProduct)); /* JTV */
  CHECK_SUNDIALS(CVodeSetPreconditioner(cvode_mem, PrecondSetup, PrecondSolve)); /* PRECOND */

  /* Adaptive GS: CGS for small problems (overhead-limited),
   * MGS for large problems (bandwidth-limited).
   * With a preconditioner, CGS is less necessary even for small
   * problems, but we keep the adaptive logic for generality.
   */
  if (neq < 500000) {
      CHECK_SUNDIALS(SUNLinSol_SPGMRSetGSType(LS, SUN_CLASSICAL_GS));
      printf("GS type: Classical (overhead-limited, neq=%d)\n", neq);
  } else {
      printf("GS type: Modified  (bandwidth-limited, neq=%d)\n", neq);
  }

  /* BDF order cap: BDF-2 reduces Nordsieck array size and
   * per-step N_VLinearCombination work.  Trade-off: may need
   * slightly more steps.  Set MAX_BDF_ORDER=5 to disable.
   */
  CHECK_SUNDIALS(CVodeSetMaxOrd(cvode_mem, MAX_BDF_ORDER));
  printf("Max BDF order: %d   Krylov dim: %d\n", MAX_BDF_ORDER, KRYLOV_DIM);

  printf("\n2D periodic smooth-texture solver (SoA) — Tier 1+3 fused (adaptive GS)\n\n");
  printf("scalar width nx = %d, rows ny = %d, groups/row = %d, ncell = %d, neq = %d\n",
         nx, ny, ng, ncell, neq);
  printf("periodic BC: x and y\n");
  printf("circle center = (%.2f, %.2f), radius = %.2f cells\n", cx, cy, radius);
  printf("smooth radial texture init\n");
  printf("core mz         = %.4f\n", (double)TEXTURE_CORE_MZ);
  printf("outer mz        = %.4f\n", (double)TEXTURE_OUTER_MZ);
  printf("width frac      = %.4f\n", (double)TEXTURE_WIDTH_FRAC);
  printf("T_TOTAL         = %.2f\n", (double)T_TOTAL);
  printf("RTOL/ATOL       = %.1e\n", (double)RTOL_VAL);
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
  Precond_Destroy(udata.pd);                   /* PRECOND */

  FusedNVec_FreePool();   /* FUSED: release persistent device buffers */

#if ENABLE_OUTPUT
  if (fp) fclose(fp);
#endif

  return 0;
}
