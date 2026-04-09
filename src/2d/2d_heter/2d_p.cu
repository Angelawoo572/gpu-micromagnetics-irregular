/**
 * 2D periodic smooth-texture LLG solver
 * CVODE + CUDA, SoA layout
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

/* Problem constants */
#define GROUPSIZE 3

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

/* user data */
typedef struct {
  int nx;      /* old scalar width = 3 * ng */
  int ny;      /* number of rows */
  int ng;      /* number of physical cells per row */
  int ncell;   /* total physical cells = ng * ny */
  int neq;     /* total equations = 3 * ncell */
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
 * RHS kernel
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

/* RHS wrapper for CVODE */
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

/* final stats */
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
  printf("nni = %-6ld ncfn = %-6ld netf = %-6ld nge = %ld\n",
         nni, ncfn, netf, nge);
}

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  SUNContext sunctx = NULL;
  sunrealtype *ydata = NULL, *abstol_data = NULL;
  sunrealtype t = T0, tout = T1;
  sunrealtype ttotal = SUN_RCONST(1000.0);

  N_Vector y = NULL, abstol = NULL;
  SUNLinearSolver LS = NULL;
  SUNNonlinearSolver NLS = NULL;
  void* cvode_mem = NULL;

  int retval, iout, NOUT;
  UserData udata;

  int cell;
#if ENABLE_OUTPUT
  int jp, ip, cell_out;
#endif

  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;

  /* problem size */
  const int nx = 900;
  const int ny = 128;

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
  const double r2 = radius * radius;
  (void)r2; /* radius still useful conceptually even though init uses rho */

  /* fill user data */
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

  /* Initialize y and abstol in SoA layout
   *
   * Smooth Néel-like radial texture:
   *   - center: approximately downward
   *   - far away: approximately upward
   *   - transition region: continuous rotation
   *
   * Construction:
   *   Let rho = distance from circle center.
   *   Define a smooth profile s(rho) in [0,1], where
   *       s ~ 1 near center
   *       s ~ 0 far outside
   *
   *   Then interpolate mz between core and outer values:
   *       mz = outer + (core - outer) * s
   *
   *   and assign the in-plane part radially:
   *       (mx, my) = m_perp * (dx, dy) / rho
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

        /* Smooth profile:
         *   s ≈ 1 deep inside
         *   s ≈ 0 far outside
         */
        const double u = (rho - radius) / width;
        const double s = 0.5 * (1.0 - tanh(u));

        /* Interpolate mz smoothly from center-down to outer-up */
        double mz0 = outer_mz + (core_mz - outer_mz) * s;

        if (mz0 >  1.0) mz0 =  1.0;
        if (mz0 < -1.0) mz0 = -1.0;

        /* in-plane magnitude from unit-length condition */
        double mperp = sqrt(fmax(0.0, 1.0 - mz0 * mz0));

        double mx0, my0;

        if (rho > (double)TEXTURE_EPS) {
          /* Néel-like radial in-plane direction */
          mx0 = mperp * (dx / rho);
          my0 = mperp * (dy / rho);
        } else {
          /* exact center: direction undefined */
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
  {
    fprintf(fp, "%f %d %d\n", (double)T0, nx, ny);
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
  printf("circle center = (%.2f, %.2f), radius = %.2f cells\n", cx, cy, radius);
  printf("smooth radial texture init\n");
  printf("core mz    = %.4f\n", (double)TEXTURE_CORE_MZ);
  printf("outer mz   = %.4f\n", (double)TEXTURE_OUTER_MZ);
  printf("width frac = %.4f\n", (double)TEXTURE_WIDTH_FRAC);

  NOUT = (int)(ttotal / T1 + SUN_RCONST(0.5));
  iout = 0;

  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start, 0));

  while (iout < NOUT) {
    retval = CVode(cvode_mem, tout, y, &t, CV_NORMAL);
    if (retval != CV_SUCCESS) {
      fprintf(stderr, "CVode error at output %d: retval = %d\n", iout, retval);
      break;
    }

#if ENABLE_OUTPUT
    if (iout % 50 == 0) {
      N_VCopyFromDevice_Cuda(y);
      ydata = N_VGetHostArrayPointer_Cuda(y);

      fprintf(fp, "%f %d %d\n", (double)t, nx, ny);
      for (jp = 0; jp < ny; jp++) {
        for (ip = 0; ip < ng; ip++) {
          cell_out = jp * ng + ip;
          fprintf(fp, "%f %f %f\n",
                  (double)ydata[idx_mx(cell_out, ncell)],
                  (double)ydata[idx_my(cell_out, ncell)],
                  (double)ydata[idx_mz(cell_out, ncell)]);
        }
      }
      fprintf(fp, "\n");
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

#if ENABLE_OUTPUT
  if (fp) fclose(fp);
#endif

  return 0;
}