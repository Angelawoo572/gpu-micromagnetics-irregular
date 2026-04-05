/**
problem: three rate equations:
   dm1/dt = m3*f2 - m2*f3 + g1 - m*g*m1
   dm2/dt = m1*f3 - m3*f1 + g2 - m*g*m2
   dm3/dt = m2*f1 - m1*f2 + g3 - m*g*m3

This program solves the problem with the BDF method using CVODE + CUDA.
Optimized version:
  - one CUDA thread per 3-component group (cell)
  - no global temporary arrays h/mh
  - no __syncthreads()
  - no cudaDeviceSynchronize() inside every RHS evaluation
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

/* Problem Constants */
#define GROUPSIZE 3
#define RTOL  SUN_RCONST(1.0e-5)
#define ATOL1 SUN_RCONST(1.0e-5)
#define ATOL2 SUN_RCONST(1.0e-5)
#define ATOL3 SUN_RCONST(1.0e-5)
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

/* simple error-checking helpers */
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
  int nx;   /* total scalar width, must be multiple of 3 */
  int ny;   /* number of rows */
  int ng;   /* number of 3-component groups per row = nx / 3 */
  int neq;  /* total number of equations = nx * ny */
} UserData;

/*
 * Right-hand-side kernel
 *
 * Mapping:
 *   one thread -> one physical 3-component group (mx,my,mz)
 *
 * Boundary policy preserved in intended form:
 *   - first/last group in x are boundary
 *   - first/last row in y are boundary
 *   - boundary derivatives set to zero
 */
__global__ static void f_kernel_group(const sunrealtype* __restrict__ y,
                                      sunrealtype* __restrict__ yd,
                                      int nx, int ny, int ng) {
  const int gx = blockIdx.x * blockDim.x + threadIdx.x;  // group index in x
  const int gy = blockIdx.y * blockDim.y + threadIdx.y;  // row index in y

  if (gx >= ng || gy >= ny) return;

  const int base = gy * nx + gx * GROUPSIZE;
  const int mx = base;
  const int my = base + 1;
  const int mz = base + 2;

  /* boundary cells: keep zero derivative */
  if (gx == 0 || gx == ng - 1 || gy == 0 || gy == ny - 1) {
    yd[mx] = ZERO;
    yd[my] = ZERO;
    yd[mz] = ZERO;
    return;
  }

  /* neighbor bases */
  const int left  = base - GROUPSIZE;
  const int right = base + GROUPSIZE;
  const int up    = base - nx;
  const int down  = base + nx;

  /* local m */
  const sunrealtype m1 = y[mx];
  const sunrealtype m2 = y[my];
  const sunrealtype m3 = y[mz];

  /*
   * Compute h for each component using the same formula as your original code:
   *
   * h[tid] = che*(left + right + up + down)
   *        + msk[comp]*(chk*m3 + cha)
   *        + chb*nsk[comp]*(left + right)
   *
   * comp=0 -> x component gets chb*(left_x + right_x)
   * comp=1 -> y component gets neither extra term
   * comp=2 -> z component gets chk*m3 + cha
   */
  const sunrealtype h1 =
      c_che * (y[left] + y[right] + y[up] + y[down]) +
      c_msk[0] * (c_chk * m3 + c_cha) +
      c_chb * c_nsk[0] * (y[left] + y[right]);

  const sunrealtype h2 =
      c_che * (y[left + 1] + y[right + 1] + y[up + 1] + y[down + 1]) +
      c_msk[1] * (c_chk * m3 + c_cha) +
      c_chb * c_nsk[1] * (y[left + 1] + y[right + 1]);

  const sunrealtype h3 =
      c_che * (y[left + 2] + y[right + 2] + y[up + 2] + y[down + 2]) +
      c_msk[2] * (c_chk * m3 + c_cha) +
      c_chb * c_nsk[2] * (y[left + 2] + y[right + 2]);

  const sunrealtype mh = m1 * h1 + m2 * h2 + m3 * h3;

  /* same cyclic cross-product-like update as original code */
  yd[mx] = c_chg * (m3 * h2 - m2 * h3) + c_alpha * (h1 - mh * m1);
  yd[my] = c_chg * (m1 * h3 - m3 * h1) + c_alpha * (h2 - mh * m2);
  yd[mz] = c_chg * (m2 * h1 - m1 * h2) + c_alpha * (h3 - mh * m3);
}

/* RHS wrapper for CVODE */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
  (void)t;  // unused

  UserData* udata = (UserData*)user_data;
  sunrealtype* ydata    = N_VGetDeviceArrayPointer_Cuda(y);
  sunrealtype* ydotdata = N_VGetDeviceArrayPointer_Cuda(ydot);

  /* 2D mapping over physical groups */
  // dim3 block(32, 8);  // 256 threads/block, warp-friendly x dimension
  // dim3 grid((udata->ng + block.x - 1) / block.x,
  //           (udata->ny + block.y - 1) / block.y);
  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid((udata->ng + block.x - 1) / block.x,
            (udata->ny + block.y - 1) / block.y);
  if (BLOCK_X * BLOCK_Y > 1024) {
    fprintf(stderr, "Invalid block size: BLOCK_X * BLOCK_Y = %d > 1024\n",
            BLOCK_X * BLOCK_Y);
    return -1;
  }

  f_kernel_group<<<grid, block>>>(ydata, ydotdata, udata->nx, udata->ny,
                                  udata->ng);

  /* launch error check only; avoid device-wide sync in hot RHS path */
  cudaError_t cuerr = cudaPeekAtLastError();
  if (cuerr != cudaSuccess) {
    fprintf(stderr, ">>> ERROR in f: kernel launch failed: %s\n",
            cudaGetErrorString(cuerr));
    return -1;
  }

  return 0;
}

/*
 * Get and print final statistics
 */
static void PrintFinalStats(void* cvode_mem, SUNLinearSolver LS) {
  (void)LS;  // unused in stats collection below

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

  int retval, iout;
  int NOUT;
  UserData udata;

  int idx, ip, jp, kp;
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;

  /* problem size */
  const int nx = 9000;
  const int ny = 128;

  if (nx % GROUPSIZE != 0) {
    fprintf(stderr, "nx must be a multiple of GROUPSIZE=%d\n", GROUPSIZE);
    return 1;
  }

  const int ng  = nx / GROUPSIZE;
  const int neq = nx * ny;

    FILE* fp = NULL;
#if ENABLE_OUTPUT
  fp = fopen("output.txt", "w");
  if (fp == NULL) {
    fprintf(stderr, "Error opening output file.\n");
    return 1;
  }
  setvbuf(fp, NULL, _IOFBF, 1 << 20);
#endif

  /* fill user data */
  udata.nx  = nx;
  udata.ny  = ny;
  udata.ng  = ng;
  udata.neq = neq;

  /* Create SUNDIALS context */
  CHECK_SUNDIALS(SUNContext_Create(SUN_COMM_NULL, &sunctx));

  /* Allocate CUDA vectors */
  y      = N_VNew_Cuda(neq, sunctx);
  abstol = N_VNew_Cuda(neq, sunctx);
  if (y == NULL || abstol == NULL) {
    fprintf(stderr, "Failed to allocate N_Vector_Cuda objects.\n");
    fclose(fp);
    return 1;
  }

  /* Get host pointers */
  ydata       = N_VGetHostArrayPointer_Cuda(y);
  abstol_data = N_VGetHostArrayPointer_Cuda(abstol);
  if (ydata == NULL || abstol_data == NULL) {
    fprintf(stderr, "Failed to get host array pointers from N_Vector_Cuda.\n");
    fclose(fp);
    N_VDestroy(y);
    N_VDestroy(abstol);
    return 1;
  }

  /* Initialize y and abstol on host */
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i += GROUPSIZE) {
      idx = i + nx * j;

      ydata[idx]     = SUN_RCONST(0.0);
      abstol_data[idx] = ATOL1;

      ydata[idx + 1]   = SUN_RCONST(0.0175);
      abstol_data[idx + 1] = ATOL2;

      if (i < nx / 2) {
        ydata[idx + 2] = SUN_RCONST(0.998);
      } else {
        ydata[idx + 2] = SUN_RCONST(-0.998);
      }
      abstol_data[idx + 2] = ATOL3;
    }
  }

  N_VCopyToDevice_Cuda(y);
  N_VCopyToDevice_Cuda(abstol);

  /* Create and initialize CVODE */
  cvode_mem = CVodeCreate(CV_BDF, sunctx);
  if (cvode_mem == NULL) {
    fprintf(stderr, "CVodeCreate failed.\n");
    goto cleanup;
  }

  CHECK_SUNDIALS(CVodeInit(cvode_mem, f, T0, y));
  CHECK_SUNDIALS(CVodeSetUserData(cvode_mem, &udata));
  CHECK_SUNDIALS(CVodeSVtolerances(cvode_mem, RTOL, abstol));

  /* nonlinear + linear solvers */
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

  printf("\nGroup of independent 3-species kinetics problems\n\n");
  printf("scalar width nx = %d, rows ny = %d, groups/row = %d, neq = %d\n",
         nx, ny, ng, neq);

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
            for (ip = 0; ip < nx - 2; ip += GROUPSIZE) {
              kp = jp * nx + ip;
              fprintf(fp, "%f %f %f\n",
                      (double)ydata[kp],
                      (double)ydata[kp + 1],
                      (double)ydata[kp + 2]);
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
  if (fp) fclose(fp);

  return 0;
}
