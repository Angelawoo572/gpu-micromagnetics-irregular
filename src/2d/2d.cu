/**
problem: three rate equations:
   dm1/dt = m3*f2 - m2*f3 + g1 - m*g*m1
   dm2/dt = m1*f3 - m3*f1 + g2 - m*g*m2
   dm3/dt = m2*f1 - m1*f2 + g3 - m*g*m3
on the interval from t = 0.0 to t = 4.e10, with
This program solves the problem with the BDF method
*/

#include <cvode/cvode.h> /* prototypes for CVODE fcts., consts.           */
#include <math.h>
#include <nvector/nvector_cuda.h> /* access to cuda N_Vector                       */
#include <stdio.h>
#include <stdlib.h>
#include <sundials/sundials_types.h> /* defs. of sunrealtype, int                        */
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>
#include <cufftdx.hpp>

// constant memory

/* Problem Constants */
#define GROUPSIZE 3 /* number of equations per group */
#define indexbound 2
#define ONE 1
#define TWO 2
#define RTOL SUN_RCONST(1.0e-5)  /* scalar relative tolerance            */
#define ATOL1 SUN_RCONST(1.0e-5) /* vector absolute tolerance components */
#define ATOL2 SUN_RCONST(1.0e-5)
#define ATOL3 SUN_RCONST(1.0e-5)
#define T0 SUN_RCONST(0.0) /* initial time           */
#define T1 SUN_RCONST(0.1) /* first output time      */
#define DT ((T1 - T0) / NOUT)
// #define NOUT      120             /* number of output times */
#define ZERO SUN_RCONST(0.0)

// constant memory
__constant__ float msk[3] = {0.0f, 0.0f, 1.0f};
__constant__ float nsk[3] = {1.0f, 0.0f, 0.0f};
__constant__ float chk = 1.0f;
__constant__ float che = 4.0f;
__constant__ float alpha = 0.2f; // 0.0f
__constant__ float chg = 1.0f;
__constant__ float cha = 0.0f; // 0.2
__constant__ float chb = 0.3f;

/* user data structure for parallel*/
typedef struct {
  int nx, ny;
  int neq; // number of equations
  sunrealtype *d_h;
  sunrealtype *d_mh;
} UserData;

/*
 *-------------------------------
 * Functions called by the solver
 *-------------------------------
 */

/* Right hand side function evaluation kernel. */
__global__ static void f_kernel(const sunrealtype *y, sunrealtype *yd,
                                sunrealtype *h, sunrealtype *mh, int nx,
                                int ny) {
  sunindextype j, k, tid, mxq, mxp, myq, myp, mx, my, mz, imsk;

  // compute 2D thread coordinates
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ix >= nx || iy >= ny)
    return;

  tid = iy * nx + ix;

  if ((ix > indexbound && ix < nx - GROUPSIZE) && (iy > 0 && iy < ny - 1)) {
    // ny - 1 since do not want the last row
    mx = tid - tid % GROUPSIZE;
    my = mx + 1;
    mz = my + 1;

    imsk = tid % GROUPSIZE;

    mxq = tid - GROUPSIZE;
    mxp = tid + GROUPSIZE;
    myq = tid - nx;
    myp = tid + nx;

    h[tid] = che * (y[mxq] + y[mxp] + y[myq] + y[myp]) +
             msk[imsk] * (chk * y[mz] + cha) +
             chb * nsk[imsk] * (y[mxq] + y[mxp]);
  }
  __syncthreads();

  if ((ix > 0 && ix < (nx - GROUPSIZE)) && (iy > 0 && iy < (ny - 1))) {
    mx = tid - tid % GROUPSIZE;
    my = mx + 1;
    mz = my + 1;

    mh[tid] = y[mx] * h[mx] + y[my] * h[my] + y[mz] * h[mz];

    int mj = (tid + 1) / GROUPSIZE;
    int nj = (tid + 2) / GROUPSIZE;
    j = tid - tid % GROUPSIZE + (tid + 1) - GROUPSIZE * mj;
    k = tid - tid % GROUPSIZE + (tid + 2) - GROUPSIZE * nj;

    yd[tid] =
        chg * (y[k] * h[j] - y[j] * h[k]) + alpha * (h[tid] - mh[tid] * y[tid]);
  } else {
    yd[tid] = 0.0;
  }
}

/* Right hand side function. This just launches the CUDA kernel
  to do the actual computation. At the very least, doing this
  saves moving the vector data in y and ydot to/from the device
  every evaluation of f. */

static int f(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data) {
  UserData *udata;
  sunrealtype *ydata, *ydotdata;

  udata = (UserData *)user_data;
  ydata = N_VGetDeviceArrayPointer_Cuda(y);
  ydotdata = N_VGetDeviceArrayPointer_Cuda(ydot);

  int nx = udata->nx, ny = udata->ny;

  int dimx = 30;
  int dimy = 32;
  dim3 block(dimx, dimy);
  int blocks_x = (nx + block.x - 1) / block.x;
  int blocks_y = (ny + block.y - 1) / block.y;
  dim3 grid(blocks_x, blocks_y);

  // printf("grid: %d %d\n",blocks_x, blocks_y);
  // printf("block: %d %d\n",block.x, block.y);
  f_kernel<<<grid, block>>>(ydata, ydotdata, udata->d_h, udata->d_mh, nx, ny);
  cudaDeviceSynchronize();

  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess) {
    fprintf(stderr, ">>> ERROR in f: cudaGetLastError returned %s\n",
            cudaGetErrorName(cuerr));
    return (-1);
  }

  return (0);
}

/*
 *-------------------------------
 * Private helper functions
 *-------------------------------
 */

/*
 * Get and print some final statistics
 */
static void PrintFinalStats(void *cvode_mem, SUNLinearSolver LS) {
  long int nst, nfe, nsetups, nni, ncfn, netf, nge;

  CVodeGetNumSteps(cvode_mem, &nst);
  CVodeGetNumRhsEvals(cvode_mem, &nfe);
  CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
  CVodeGetNumErrTestFails(cvode_mem, &netf);
  CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
  CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
  CVodeGetNumGEvals(cvode_mem, &nge);

  printf("\nFinal Statistics:\n");
  printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld", nst, nfe, nsetups);
  printf("nni = %-6ld ncfn = %-6ld netf = %-6ld    nge = %ld\n", nni, ncfn,
         netf, nge);
}

/*
 *-------------------------------
 * Main Program
 *-------------------------------
 */
int main(int argc, char *argv[]) {
  SUNContext sunctx; // SUNDIALS context
  sunrealtype *ydata,
      *abstol_data; // Host-side pointers to solution and tolerance data
  sunrealtype t;
  sunrealtype tout;
  N_Vector y,
      abstol; // SUNDIALS vector structures for solution and absolute tolerance
  SUNLinearSolver LS; // Linear solver object (cuSolverSp QR)
  SUNNonlinearSolver NLS;
  void *cvode_mem;  // CVODE integrator memory
  int retval, iout; // return status and output counter
  int neq; // Problem size: number of equations, groups, and loop index
  UserData udata;
  int idx;
  int ip, jp, kp;
  cudaEvent_t start, stop;
  float elapsedTime;

  /* Parse command-line to get number of groups */
  int nx = 9000, ny = 128;
  neq = nx * ny;

  FILE *fp = fopen("output.txt", "w");
  if (fp == NULL) {
    fprintf(stderr, "Error opening output file.\n");
    return 1;
  }

  /* Fill user data */
  udata.nx = nx;
  udata.ny = ny;
  udata.neq = neq;
  cudaMalloc(&udata.d_h, neq * sizeof(sunrealtype));
  cudaMalloc(&udata.d_mh, neq * sizeof(sunrealtype));

  /* Create SUNDIALS context */
  SUNContext_Create(SUN_COMM_NULL, &sunctx);

  /* Allocate CUDA vectors for solution and tolerances */
  y = N_VNew_Cuda(neq, sunctx);
  abstol = N_VNew_Cuda(neq, sunctx);
  // get host pointers
  ydata = N_VGetHostArrayPointer_Cuda(y);
  abstol_data = N_VGetHostArrayPointer_Cuda(abstol);

  /* Initialize y and abstol on host then copy to device */
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i += 3) {
      idx = i + nx * j;

      ydata[idx] = 0.0;
      abstol_data[idx] = ATOL1;

      ydata[idx + 1] = 0.0175;
      abstol_data[idx + 1] = ATOL2;
      if (i < nx / 2) {
        ydata[idx + 2] = 0.998;
      } else {
        ydata[idx + 2] = -0.998;
      }
      abstol_data[idx + 2] = ATOL3;
    }
  }

  N_VCopyToDevice_Cuda(y);
  N_VCopyToDevice_Cuda(abstol);

  /* Create and initialize CVODE solver memory */
  cvode_mem = CVodeCreate(CV_BDF, sunctx);
  CVodeInit(cvode_mem, f, T0, y);
  CVodeSetUserData(cvode_mem, &udata);
  CVodeSVtolerances(cvode_mem, RTOL, abstol);

  /* Matrix-free GMRES linear solver (no Jacobian needed) */
  NLS = SUNNonlinSol_Newton(y, sunctx);
  CVodeSetNonlinearSolver(cvode_mem, NLS);
  LS = SUNLinSol_SPGMR(y, SUN_PREC_NONE, 0, sunctx);
  CVodeSetLinearSolver(cvode_mem, LS, NULL);

  /* Print header */
  printf("\nGroup of independent 3-species kinetics problems\n\n");
  printf("number of groups = %d %d %d \n", nx, ny, nx * ny);

  /* Time-stepping loop */

  float ttotal = 1000.0f;
  iout = T0;
  tout = T1;
  int NOUT = ttotal / T1;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //calculate time
  // print output
  while (iout < NOUT) {

    retval = CVode(cvode_mem, tout, y, &t, CV_NORMAL);

    // copy solution back to host and print all groups

    if (retval != CV_SUCCESS) {
      fprintf(stderr, "CVode error at output %d: retval = %d\n", iout, retval);
      break;
    }

    // N_VCopyFromDevice_Cuda(y);
    // ydata = N_VGetHostArrayPointer_Cuda(y);

    if (iout % 50 == 0) {
      N_VCopyFromDevice_Cuda(y);
      ydata = N_VGetHostArrayPointer_Cuda(y);
      fprintf(fp,"%f %d %d \n", t, nx, ny);
      for (jp = 0; jp < ny; jp++) {
        for (ip = 0; ip < nx - 2; ip += 3) {
          kp = jp * nx + ip;
          fprintf(fp, "%f %f %f\n", ydata[kp], ydata[kp + 1], ydata[kp + 2]);
        }
      }
      fprintf(fp, "\n");
    }

    iout++;
    tout += T1;
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("GPU simulation took %.3f ms\n", elapsedTime);

  /* Print final statistics */
  PrintFinalStats(cvode_mem, LS);

  /* Clean up */
  cudaFree(udata.d_h);
  cudaFree(udata.d_mh);
  N_VDestroy(y);
  N_VDestroy(abstol);
  CVodeFree(&cvode_mem);
  SUNLinSolFree(LS);
  SUNNonlinSolFree(NLS);
  SUNContext_Free(&sunctx);
  fclose(fp);

  return 0;
}
