/** 
problem: three rate equations:
    dm1/dt = m3*f2 - m2*f3 + g1 - m*g*m1
    dm2/dt = m1*f3 - m3*f1 + g2 - m*g*m2
    dm3/dt = m2*f1 - m1*f2 + g3 - m*g*m3
on the interval from t = 0.0 to t = 4.e10, with 
This program solves the problem with the BDF method
*/

#include <cvode/cvode.h> /* prototypes for CVODE fcts., consts.           */
#include <nvector/nvector_cuda.h> /* access to cuda N_Vector                       */
#include <stdio.h>
#include <stdlib.h>
#include <sundials/sundials_types.h> /* defs. of sunrealtype, int                        */
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>
#include <math.h>

// constant memory

/* Problem Constants */ 
#define GROUPSIZE 3               /* number of equations per group */
#define indexbound 2
#define ONE 1
#define TWO 2
#define RTOL      SUN_RCONST(1.0e-5) /* scalar relative tolerance            */
#define ATOL1     SUN_RCONST(1.0e-5) /* vector absolute tolerance components */
#define ATOL2     SUN_RCONST(1.0e-5)
#define ATOL3     SUN_RCONST(1.0e-5)
#define T0        SUN_RCONST(0.0)  /* initial time           */
#define T1        SUN_RCONST(0.1)  /* first output time      */
#define DT    ((T1 - T0) / NOUT)
// #define NOUT      120             /* number of output times */

#define ZERO SUN_RCONST(0.0)

// constant memory
__constant__ float msk[3]={0.0f,0.0f,1.0f};
__constant__ float nsk[3]={1.0f,0.0f,0.0f};
__constant__ float chk=1.0f;
__constant__ float che =0.0f;
__constant__ float alpha=0.02f;  // 0.0f
__constant__ float chg = 1.0f; 
__constant__ float cha = 1.5f; //0.2
__constant__ float chb = 0.0f;

/* user data structure for parallel*/
typedef struct
{
  int ngroups; // number of groups
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
__global__ static void f_kernel(
  const sunrealtype* y, 
  sunrealtype* yd, 
  sunrealtype* h,
  sunrealtype* mh,
  int neq)
{
  sunindextype i, j, k, tid,iq,ip,ix,iy,iz,imsk;
  // thread index
  tid = blockDim.x * blockIdx.x + threadIdx.x;
  if ( tid > indexbound && tid < blockDim.x - GROUPSIZE){
    iq = tid - GROUPSIZE; // 前一组位置, -3
    ip = tid + GROUPSIZE; // 后一组位置, +3
    ix = tid - (tid) % GROUPSIZE; // ix = 3 * (tid / 3)
    iy = ix + ONE;
    iz = iy + ONE;
    imsk = tid % GROUPSIZE; // tid在3个一组的thread的相对位置 x = 0, y = 1, z = 2
    /*
    normalize effective field, vector f
    che*(y[iq]+y[ip]); exchange interaction
    msk[imsk]*chk*y[iz]; AnisotropyTrem
     */
    h[tid] = che*(y[iq]+y[ip])+msk[imsk]*(chk*y[iz]+cha)+nsk[imsk]*(y[ix+3]+y[ix-3])*chb;
    // printf("h[%d] = %g\n", tid, che*(y[iq]+y[ip]) + msk[imsk]*chk*y[iz]);
  }
  __syncthreads();
  if ( tid > indexbound && tid < blockDim.x - GROUPSIZE){
    i = tid - tid % GROUPSIZE; // x
    j = i + ONE; // y
    k = j + ONE; // x
    // m 点乘 f,3个维度 dot product
    mh[tid]=y[i]*h[i]+y[j]*h[j]+y[k]*h[k];

    // j=tid+(tid+1)%3;
    int M = (tid+ONE) /GROUPSIZE;
    int N = (tid + TWO) / GROUPSIZE;
    j = ( tid - tid % GROUPSIZE) + (tid + ONE) - GROUPSIZE * M;
    k = (tid - tid % GROUPSIZE) + (tid + TWO) - GROUPSIZE * N;
    // k=tid+(tid+2)%3;
    /* 
    g = alpha * f
    dm/dtao = m叉乘f 前一部分 cross product
    y[tid] is m
    */
    yd[tid] = chg*(y[k]*h[j] - y[j]*h[k]) + alpha*(h[tid] - mh[tid]*y[tid]);
  }
  else
  {
    yd[tid] = 0;
    // printf("DEBUG: entering kernel, neq = %d\n", neq);
  }
   __syncthreads();
}

/* Right hand side function. This just launches the CUDA kernel
   to do the actual computation. At the very least, doing this
   saves moving the vector data in y and ydot to/from the device
   every evaluation of f. */

static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
    UserData* udata;
    sunrealtype *ydata, *ydotdata;

    udata    = (UserData*)user_data;
    ydata    = N_VGetDeviceArrayPointer_Cuda(y);
    ydotdata = N_VGetDeviceArrayPointer_Cuda(ydot);

    unsigned block_size = GROUPSIZE * 32;
    // total threads = grid_size * block_size
    // grid_size is ceil - (a+b-1)/b
    unsigned grid_size  = 1; // 1 (udata->neq + block_size - 1) / block_size
    f_kernel<<<grid_size, block_size>>>(ydata, ydotdata,udata->d_h,
      udata->d_mh, udata->neq);

    cudaDeviceSynchronize();

    //debug
    // sunrealtype h_ydot[9];
    // cudaMemcpy(h_ydot, ydotdata + 3, 3 * sizeof(sunrealtype), cudaMemcpyDeviceToHost);
    // printf("ydot sample (group 1): %f %f %f\n", h_ydot[0], h_ydot[1], h_ydot[2]);
    
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
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
static void PrintOutput(sunrealtype t, sunrealtype y1, sunrealtype y2,
                        sunrealtype y3)
{
#if defined(SUNDIALS_EXTENDED_PRECISION)
  printf("At t = %0.4Le      y =%14.6Le  %14.6Le  %14.6Le\n", t, y1, y2, y3);
#elif defined(SUNDIALS_DOUBLE_PRECISION)
  printf("At t = %0.4e      y =%14.6e  %14.6e  %14.6e\n", t, y1, y2, y3);
#else
  printf("At t = %0.4e      y =%14.6e  %14.6e  %14.6e\n", t, y1, y2, y3);
#endif

  return;
}

/*
 * Get and print some final statistics
 */
static void PrintFinalStats(void* cvode_mem, SUNLinearSolver LS)
{
  long int nst, nfe, nsetups, nni, ncfn, netf, nge;

  CVodeGetNumSteps(cvode_mem, &nst);
  CVodeGetNumRhsEvals(cvode_mem, &nfe);
  CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
  CVodeGetNumErrTestFails(cvode_mem, &netf);
  CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
  CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
  CVodeGetNumGEvals(cvode_mem, &nge);


  printf("\nFinal Statistics:\n");
  printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld", nst, nfe,
         nsetups);
  printf("nni = %-6ld ncfn = %-6ld netf = %-6ld    nge = %ld\n", nni, ncfn,
         netf, nge);
}

/*
 *-------------------------------
 * Main Program
 *-------------------------------
 */
int main(int argc, char* argv[])
{
    SUNContext sunctx; // SUNDIALS context
    sunrealtype *ydata, *abstol_data; // Host-side pointers to solution and tolerance data
    sunrealtype t;
    sunrealtype tout;
    N_Vector y, abstol; // SUNDIALS vector structures for solution and absolute tolerance
    SUNLinearSolver LS; // Linear solver object (cuSolverSp QR)
    SUNNonlinearSolver NLS;
    void* cvode_mem; // CVODE integrator memory
    int retval, iout; // return status and output counter
    int neq, ngroups, groupj;// Problem size: number of equations, groups, and loop index
    UserData udata;

    /* Parse command-line to get number of groups */
    ngroups = 32;
    neq     = ngroups * GROUPSIZE;

    /* Fill user data */
    udata.ngroups = ngroups;
    udata.neq     = neq;
    cudaMalloc(&udata.d_h,  neq * sizeof(sunrealtype));
    cudaMalloc(&udata.d_mh, neq * sizeof(sunrealtype));

    /* Create SUNDIALS context */
    SUNContext_Create(SUN_COMM_NULL, &sunctx);

    /* Allocate CUDA vectors for solution and tolerances */
    y     = N_VNew_Cuda(neq, sunctx);
    abstol= N_VNew_Cuda(neq, sunctx);
    // get host pointers
    ydata       = N_VGetHostArrayPointer_Cuda(y);
    abstol_data = N_VGetHostArrayPointer_Cuda(abstol);

    /* Initialize y and abstol on host then copy to device */
    int nspin = ngroups; //32
    int ix, iy, iz;

    for(int i = 0;i < nspin;i++)
    {
	    ix=3*i;
	    iy=ix+1;
	    iz=iy+1;

	    if(i==0)
	    {
		    ydata[ix]=0.0;
		    ydata[iy]=0.0;
		    ydata[iz]=1.0;
	    }
	    else if(i == nspin-1)
	    {
		    ydata[ix]=0.0;
		    ydata[iy]=0.0;
		    ydata[iz]=-1;;
	    }
	    else if(i < nspin/2)
	    {
		    ydata[ix]=0.0;
		    ydata[iy]=0.0175;
		    ydata[iz]=0.998;
	    }
	    else
	    {
		    ydata[ix]=0.0;
		    ydata[iy]=0.0175;
		    ydata[iz]=-0.998;
      }
    }

    for (int i = 0; i < neq; i += 3) {
        abstol_data[i]   = ATOL1;
        abstol_data[i+1] = ATOL2;
        abstol_data[i+2] = ATOL3;
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
    printf("number of groups = %d\n\n", ngroups);

    /* Time-stepping loop */
    float ttotal=500.0f;
    iout = T0;
    tout = T1;
    int NOUT=ttotal/T1;
    while (iout < NOUT) {
      // &t cvode实际走到的地方
      retval = CVode(cvode_mem, tout, y, &t, CV_NORMAL);
      // copy solution back to host and print all groups
      if (retval == CV_SUCCESS) {
        iout++;
        tout += T1; // T0 + iout*T1
      }else {
        fprintf(stderr, "CVode error at output %d: retval = %d\n", iout, retval);
        break;
      }
      // printf("%f\n",tout);
    }
    N_VCopyFromDevice_Cuda(y);
    ydata = N_VGetHostArrayPointer_Cuda(y);
    printf("\n=== Old constants final t ===\n");
    for (groupj = 0; groupj < ngroups; groupj ++) {
      printf("group %d: ", groupj);
      PrintOutput(t,ydata[GROUPSIZE * groupj],
                    ydata[1 + GROUPSIZE * groupj],
                    ydata[2 + GROUPSIZE * groupj]);
    }

    // 把 host 端新值拷到 GPU constant memory
    float host_cha   = -0.6f;
    float host_alpha = 0.0f;
    cudaMemcpyToSymbol(cha,   &host_cha,   sizeof(float));
    // cudaMemcpyToSymbol(alpha, &host_alpha, sizeof(float));
    CVodeReInit(cvode_mem, T0, y);
    iout = 0;
    tout = T1;

    printf("\n=== New constants, printing every time step ===\n");
    while (iout < NOUT) {
      retval = CVode(cvode_mem, tout, y, &t, CV_NORMAL);
      if (retval != CV_SUCCESS) {
        fprintf(stderr, "CVode error at step %d: %d\n", iout, retval);
        break;
      }

      // 每一步都把解拷回，并打印所有 group
      N_VCopyFromDevice_Cuda(y);
      ydata = N_VGetHostArrayPointer_Cuda(y);
      // printf("t = %0.4e\n", t);
      if (iout % 1 == 0) {
        for (int gj = 20; gj < 21; gj++) {
        printf("  group %2d: ", gj);
        PrintOutput(t,
                    ydata[3*gj + 0],
                    ydata[3*gj + 1],
                    ydata[3*gj + 2]);
      }
      }

      iout++;
      tout += T1;
    }

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

    return 0;
}
