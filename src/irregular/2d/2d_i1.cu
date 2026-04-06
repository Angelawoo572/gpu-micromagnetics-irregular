/**
 * 2D irregular compact active-cell execution for CVODE + CUDA (SoA)
 *
 * Geometry:
 *   - periodic in x
 *   - periodic in y
 *   - one circular masked hole in the interior
 *
 * Execution policy:
 *   - compact active-cell launch for RHS
 *   - one CUDA thread per active cell
 *   - inactive cells are stored in the full state vector but do not evolve
 *   - inactive entries in ydot are explicitly zeroed each RHS call
 *   - neighbor contributions from inactive cells are skipped
 *
 * Notes:
 *   - State layout remains dense SoA in one N_Vector:
 *       [mx for all cells][my for all cells][mz for all cells]
 *   - This isolates the effect of execution compaction without changing the
 *     mathematical state representation expected by CVODE.
 */

#include <cvode/cvode.h>
#include <cuda_runtime.h>
#include <math.h>
#include <nvector/nvector_cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>
#include <sundials/sundials_types.h>

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

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef HOLE_CENTER_X_FRAC
#define HOLE_CENTER_X_FRAC 0.50
#endif

#ifndef HOLE_CENTER_Y_FRAC
#define HOLE_CENTER_Y_FRAC 0.50
#endif

#ifndef HOLE_RADIUS_FRAC_Y
#define HOLE_RADIUS_FRAC_Y 0.22
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
  int nx;
  int ny;
  int ng;
  int ncell;
  int neq;

  unsigned char* h_active;
  unsigned char* d_active;

  int* h_active_ids;
  int* d_active_ids;
  int  n_active;

  int* h_inactive_ids;
  int* d_inactive_ids;
  int  n_inactive;
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

/* zero ydot only for inactive cells */
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

/* compact RHS kernel: one thread per active cell */
__global__ static void f_kernel_group_soa_irregular_compact(
    const sunrealtype* __restrict__ y,
    sunrealtype* __restrict__ yd,
    const unsigned char* __restrict__ active,
    const int* __restrict__ active_ids,
    int n_active,
    int ng, int ny, int ncell) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_active) return;

  const int cell = active_ids[tid];
  const int gx = cell % ng;
  const int gy = cell / ng;

  const int mx = idx_mx(cell, ncell);
  const int my = idx_my(cell, ncell);
  const int mz = idx_mz(cell, ncell);

  const int xl = wrap_x(gx - 1, ng);
  const int xr = wrap_x(gx + 1, ng);
  const int yu = wrap_y(gy - 1, ny);
  const int ydwn = wrap_y(gy + 1, ny);

  const int left_cell  = gy * ng + xl;
  const int right_cell = gy * ng + xr;
  const int up_cell    = yu * ng + gx;
  const int down_cell  = ydwn * ng + gx;

  const sunrealtype m1 = y[mx];
  const sunrealtype m2 = y[my];
  const sunrealtype m3 = y[mz];

  sunrealtype sum_x_h1 = ZERO, sum_y_h1 = ZERO;
  sunrealtype sum_x_h2 = ZERO, sum_y_h2 = ZERO;
  sunrealtype sum_x_h3 = ZERO, sum_y_h3 = ZERO;
  sunrealtype sum_lr_h1 = ZERO, sum_lr_h2 = ZERO, sum_lr_h3 = ZERO;

  if (active[left_cell]) {
    sum_x_h1 += y[idx_mx(left_cell, ncell)];
    sum_x_h2 += y[idx_my(left_cell, ncell)];
    sum_x_h3 += y[idx_mz(left_cell, ncell)];

    sum_lr_h1 += y[idx_mx(left_cell, ncell)];
    sum_lr_h2 += y[idx_my(left_cell, ncell)];
    sum_lr_h3 += y[idx_mz(left_cell, ncell)];
  }

  if (active[right_cell]) {
    sum_x_h1 += y[idx_mx(right_cell, ncell)];
    sum_x_h2 += y[idx_my(right_cell, ncell)];
    sum_x_h3 += y[idx_mz(right_cell, ncell)];

    sum_lr_h1 += y[idx_mx(right_cell, ncell)];
    sum_lr_h2 += y[idx_my(right_cell, ncell)];
    sum_lr_h3 += y[idx_mz(right_cell, ncell)];
  }

  if (active[up_cell]) {
    sum_y_h1 += y[idx_mx(up_cell, ncell)];
    sum_y_h2 += y[idx_my(up_cell, ncell)];
    sum_y_h3 += y[idx_mz(up_cell, ncell)];
  }

  if (active[down_cell]) {
    sum_y_h1 += y[idx_mx(down_cell, ncell)];
    sum_y_h2 += y[idx_my(down_cell, ncell)];
    sum_y_h3 += y[idx_mz(down_cell, ncell)];
  }

  const sunrealtype h1 =
      c_che * (sum_x_h1 + sum_y_h1) +
      c_msk[0] * (c_chk * m3 + c_cha) +
      c_chb * c_nsk[0] * sum_lr_h1;

  const sunrealtype h2 =
      c_che * (sum_x_h2 + sum_y_h2) +
      c_msk[1] * (c_chk * m3 + c_cha) +
      c_chb * c_nsk[1] * sum_lr_h2;

  const sunrealtype h3 =
      c_che * (sum_x_h3 + sum_y_h3) +
      c_msk[2] * (c_chk * m3 + c_cha) +
      c_chb * c_nsk[2] * sum_lr_h3;

  const sunrealtype mh = m1 * h1 + m2 * h2 + m3 * h3;

  yd[mx] = c_chg * (m3 * h2 - m2 * h3) + c_alpha * (h1 - mh * m1);
  yd[my] = c_chg * (m1 * h3 - m3 * h1) + c_alpha * (h2 - mh * m2);
  yd[mz] = c_chg * (m2 * h1 - m1 * h2) + c_alpha * (h3 - mh * m3);
}

/* RHS wrapper */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
  (void)t;

  UserData* udata = (UserData*)user_data;
  sunrealtype* ydata    = N_VGetDeviceArrayPointer_Cuda(y);
  sunrealtype* ydotdata = N_VGetDeviceArrayPointer_Cuda(ydot);

  if (BLOCK_SIZE > 1024) {
    fprintf(stderr, "Invalid BLOCK_SIZE=%d > 1024\n", BLOCK_SIZE);
    return -1;
  }

  /* zero inactive entries each RHS call */
  if (udata->n_inactive > 0) {
    int grid0 = (udata->n_inactive + BLOCK_SIZE - 1) / BLOCK_SIZE;
    zero_inactive_kernel<<<grid0, BLOCK_SIZE>>>(
        ydotdata, udata->d_inactive_ids, udata->n_inactive, udata->ncell);
  }

  /* compact active-cell launch */
  if (udata->n_active > 0) {
    int grid1 = (udata->n_active + BLOCK_SIZE - 1) / BLOCK_SIZE;
    f_kernel_group_soa_irregular_compact<<<grid1, BLOCK_SIZE>>>(
        ydata, ydotdata, udata->d_active, udata->d_active_ids,
        udata->n_active, udata->ng, udata->ny, udata->ncell);
  }

  cudaError_t cuerr = cudaPeekAtLastError();
  if (cuerr != cudaSuccess) {
    fprintf(stderr, ">>> ERROR in f: kernel launch failed: %s\n",
            cudaGetErrorString(cuerr));
    return -1;
  }

  return 0;
}

/* Final stats */
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

/* Build mask and compact lists on host */
static void BuildCircularHoleMaskAndLists(UserData* udata) {
  const int ng = udata->ng;
  const int ny = udata->ny;

  const double cx = HOLE_CENTER_X_FRAC * (double)(ng - 1);
  const double cy = HOLE_CENTER_Y_FRAC * (double)(ny - 1);
  const double radius = HOLE_RADIUS_FRAC_Y * (double)ny;
  const double r2 = radius * radius;

  int n_active = 0;
  int n_inactive = 0;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < ng; ++i) {
      const double dx = (double)i - cx;
      const double dy = (double)j - cy;
      const double dist2 = dx * dx + dy * dy;
      const int cell = j * ng + i;

      if (dist2 <= r2) {
        udata->h_active[cell] = 0;
        n_inactive++;
      } else {
        udata->h_active[cell] = 1;
        n_active++;
      }
    }
  }

  udata->n_active = n_active;
  udata->n_inactive = n_inactive;

  udata->h_active_ids = (int*)malloc((size_t)n_active * sizeof(int));
  udata->h_inactive_ids = (int*)malloc((size_t)n_inactive * sizeof(int));

  if ((n_active > 0 && udata->h_active_ids == NULL) ||
      (n_inactive > 0 && udata->h_inactive_ids == NULL)) {
    fprintf(stderr, "Failed to allocate compact index arrays.\n");
    exit(EXIT_FAILURE);
  }

  int ia = 0, ii = 0;
  for (int cell = 0; cell < udata->ncell; ++cell) {
    if (udata->h_active[cell]) {
      udata->h_active_ids[ia++] = cell;
    } else {
      udata->h_inactive_ids[ii++] = cell;
    }
  }
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

  const int nx = 1536;
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

  udata.nx = nx;
  udata.ny = ny;
  udata.ng = ng;
  udata.ncell = ncell;
  udata.neq = neq;
  udata.h_active = NULL;
  udata.d_active = NULL;
  udata.h_active_ids = NULL;
  udata.d_active_ids = NULL;
  udata.h_inactive_ids = NULL;
  udata.d_inactive_ids = NULL;
  udata.n_active = 0;
  udata.n_inactive = 0;

  CHECK_SUNDIALS(SUNContext_Create(SUN_COMM_NULL, &sunctx));

  udata.h_active = (unsigned char*)malloc((size_t)ncell * sizeof(unsigned char));
  if (udata.h_active == NULL) {
    fprintf(stderr, "Failed to allocate host active mask.\n");
    goto cleanup;
  }

  BuildCircularHoleMaskAndLists(&udata);

  CHECK_CUDA(cudaMalloc((void**)&udata.d_active,
                        (size_t)ncell * sizeof(unsigned char)));

  if (udata.n_active > 0) {
    CHECK_CUDA(cudaMalloc((void**)&udata.d_active_ids,
                          (size_t)udata.n_active * sizeof(int)));
  }
  if (udata.n_inactive > 0) {
    CHECK_CUDA(cudaMalloc((void**)&udata.d_inactive_ids,
                          (size_t)udata.n_inactive * sizeof(int)));
  }

  CHECK_CUDA(cudaMemcpy(udata.d_active, udata.h_active,
                        (size_t)ncell * sizeof(unsigned char),
                        cudaMemcpyHostToDevice));

  if (udata.n_active > 0) {
    CHECK_CUDA(cudaMemcpy(udata.d_active_ids, udata.h_active_ids,
                          (size_t)udata.n_active * sizeof(int),
                          cudaMemcpyHostToDevice));
  }
  if (udata.n_inactive > 0) {
    CHECK_CUDA(cudaMemcpy(udata.d_inactive_ids, udata.h_inactive_ids,
                          (size_t)udata.n_inactive * sizeof(int),
                          cudaMemcpyHostToDevice));
  }

  printf("\n2D irregular compact (SoA, compact active-cell execution)\n\n");
  printf("scalar width nx = %d, rows ny = %d, groups/row = %d, ncell = %d, neq = %d\n",
         nx, ny, ng, ncell, neq);
  printf("periodic BC: x and y\n");
  printf("masked hole: center=(%.2f, %.2f), radius=%.2f cells\n",
         HOLE_CENTER_X_FRAC * (double)(ng - 1),
         HOLE_CENTER_Y_FRAC * (double)(ny - 1),
         HOLE_RADIUS_FRAC_Y * (double)ny);
  printf("active cells   = %d / %d (%.2f%% active)\n",
         udata.n_active, ncell,
         100.0 * (double)udata.n_active / (double)ncell);
  printf("inactive cells = %d / %d (%.2f%% inactive)\n",
         udata.n_inactive, ncell,
         100.0 * (double)udata.n_inactive / (double)ncell);

  y      = N_VNew_Cuda(neq, sunctx);
  abstol = N_VNew_Cuda(neq, sunctx);
  if (y == NULL || abstol == NULL) {
    fprintf(stderr, "Failed to allocate N_Vector_Cuda objects.\n");
    goto cleanup;
  }

  ydata       = N_VGetHostArrayPointer_Cuda(y);
  abstol_data = N_VGetHostArrayPointer_Cuda(abstol);
  if (ydata == NULL || abstol_data == NULL) {
    fprintf(stderr, "Failed to get host array pointers from N_Vector_Cuda.\n");
    goto cleanup;
  }

  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < ng; i++) {
      cell = j * ng + i;

      const int mx = idx_mx(cell, ncell);
      const int my = idx_my(cell, ncell);
      const int mz = idx_mz(cell, ncell);

      if (udata.h_active[cell]) {
        ydata[mx] = SUN_RCONST(0.0);
        ydata[my] = SUN_RCONST(0.0175);
        ydata[mz] = (i < ng / 2) ? SUN_RCONST(0.998) : SUN_RCONST(-0.998);
      } else {
        ydata[mx] = ZERO;
        ydata[my] = ZERO;
        ydata[mz] = ZERO;
      }

      abstol_data[mx] = ATOL1;
      abstol_data[my] = ATOL2;
      abstol_data[mz] = ATOL3;
    }
  }

  N_VCopyToDevice_Cuda(y);
  N_VCopyToDevice_Cuda(abstol);

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
          if (udata.h_active[cell_out]) {
            fprintf(fp, "%f %f %f\n",
                    (double)ydata[idx_mx(cell_out, ncell)],
                    (double)ydata[idx_my(cell_out, ncell)],
                    (double)ydata[idx_mz(cell_out, ncell)]);
          } else {
            fprintf(fp, "0.0 0.0 0.0\n");
          }
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
  if (udata.d_active) cudaFree(udata.d_active);
  if (udata.d_active_ids) cudaFree(udata.d_active_ids);
  if (udata.d_inactive_ids) cudaFree(udata.d_inactive_ids);

  if (udata.h_active) free(udata.h_active);
  if (udata.h_active_ids) free(udata.h_active_ids);
  if (udata.h_inactive_ids) free(udata.h_inactive_ids);

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