/**
 * 2D periodic smooth-texture LLG solver — with FFT Demagnetization Field
 *
 * NEW: h_total = h_exchange + h_demag
 *      h_demag(i,j) = IFFT[ D̂(kx,ky) · M̂(kx,ky) ]   (convolution theorem)
 *
 * From the handwritten derivation:
 *   h_dmag(i,j) = Σ_{m,n} D(i-m, j-n) · M(m,n)      [space]
 *               = IFFT[ D̂ · M̂ ]                       [k-space, O(N log N)]
 *
 * Controlled by DEMAG_STRENGTH (default 0.0 = off, 1.0 = full physics).
 *
 * All other optimizations unchanged:
 *   - Tier 1 + Tier 3 fused NVector ops (deferred_nvector.cu)
 *   - 3x3 block Jacobi preconditioner   (precond.cu)
 *   - Analytic Jacobian-times-vector     (jtv.cu)
 *   - FFT demagnetization field          (fft_demag.cu)  <<< NEW
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
#include <sundials/sundials_iterative.h>

#include "deferred_nvector.h"
#include "precond.h"
#include "jtv.h"
#include "fft_demag.h"   /* <<< FFT DEMAG: h = IFFT[D̂·M̂] */

/* Problem constants */
#define GROUPSIZE 3

/* -------------------------------------------------------
 * Solver tuning knobs
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

/* FFT Demag strength: 0=disabled, 1=full dipolar physics */
#ifndef DEMAG_STRENGTH
#define DEMAG_STRENGTH 0.0
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
#ifndef CIRCLE_RADIUS_FRAC_Y
#define CIRCLE_RADIUS_FRAC_Y 0.22
#endif

#ifndef TEXTURE_CORE_MZ
#define TEXTURE_CORE_MZ -0.998
#endif
#ifndef TEXTURE_OUTER_MZ
#define TEXTURE_OUTER_MZ 0.998
#endif
#ifndef TEXTURE_WIDTH_FRAC
#define TEXTURE_WIDTH_FRAC 0.35
#endif
#ifndef TEXTURE_EPS
#define TEXTURE_EPS 1.0e-12
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
 * IMPORTANT: PrecondData *pd MUST be the first field (offset 0).
 */
typedef struct {
  PrecondData  *pd;      /* first field: 3x3 block Jacobi preconditioner   */
  DemagData    *demag;   /* <<< FFT DEMAG: cuFFT dipolar field handle      */
  sunrealtype  *d_hdmag; /* <<< FFT DEMAG: device SoA buffer, 3*ncell     */
  int nx, ny, ng, ncell, neq;
} UserData;

/* SoA indexing helpers */
__host__ __device__ static inline int idx_mx(int cell, int ncell) { return cell; }
__host__ __device__ static inline int idx_my(int cell, int ncell) { return ncell + cell; }
__host__ __device__ static inline int idx_mz(int cell, int ncell) { return 2 * ncell + cell; }
__host__ __device__ static inline int wrap_x(int x, int ng) {
  return (x < 0) ? (x + ng) : ((x >= ng) ? (x - ng) : x);
}
__host__ __device__ static inline int wrap_y(int y, int ny) {
  return (y < 0) ? (y + ny) : ((y >= ny) ? (y - ny) : y);
}

/* -----------------------------------------------------------------------
 * Exchange RHS kernel — unchanged from baseline
 * --------------------------------------------------------------------- */
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

/* -----------------------------------------------------------------------
 * Demag correction kernel  (NEW)
 *
 * Adds the LLG contribution of h_dmag to ydot:
 *   Δ(dm/dt) = γ (m × h_dmag) + α (h_dmag - (m·h_dmag)·m)
 *
 * Math matches the handwritten derivation:
 *   h_dmag(i,j) = IFFT[ D̂(k) · M̂(k) ]
 *   this kernel applies LLG(m, h_dmag) as additive correction to ydot
 * --------------------------------------------------------------------- */
__global__ static void demag_correction_kernel(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ h_dmag,
    sunrealtype*       __restrict__ ydot,
    int ncell)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= ncell) return;

    const sunrealtype m1 = y[cell];
    const sunrealtype m2 = y[ncell + cell];
    const sunrealtype m3 = y[2*ncell + cell];

    const sunrealtype hd1 = h_dmag[cell];
    const sunrealtype hd2 = h_dmag[ncell + cell];
    const sunrealtype hd3 = h_dmag[2*ncell + cell];

    /* m · h_dmag (needed for damping term) */
    const sunrealtype mhd = m1*hd1 + m2*hd2 + m3*hd3;

    /* LLG additive correction:
     *   Δf_α = γ (m × hd)_α + α (hd_α - (m·hd) m_α)
     */
    ydot[cell]          += c_chg*(m3*hd2 - m2*hd3) + c_alpha*(hd1 - mhd*m1);
    ydot[ncell + cell]  += c_chg*(m1*hd3 - m3*hd1) + c_alpha*(hd2 - mhd*m2);
    ydot[2*ncell + cell]+= c_chg*(m2*hd1 - m1*hd2) + c_alpha*(hd3 - mhd*m3);
}

/* -----------------------------------------------------------------------
 * RHS wrapper for CVODE
 *
 * Total effective field:
 *   h_total = h_exchange   (nearest-neighbor stencil, original)
 *           + h_demag      (long-range dipolar, FFT-based, NEW)
 *
 * h_dmag(i,j) = Σ_{m,n} D(i-m, j-n) · M(m,n)   [convolution]
 *             = IFFT[ D̂(kx,ky) · M̂(kx,ky) ]    [via cuFFT, O(N log N)]
 * --------------------------------------------------------------------- */
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

  /* Step 1: Exchange stencil + anisotropy (original RHS kernel) */
  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid((udata->ng + block.x - 1) / block.x,
            (udata->ny + block.y - 1) / block.y);

  f_kernel_group_soa_periodic<<<grid, block>>>(
      ydata, ydotdata, udata->ng, udata->ny, udata->ncell);

  /* Step 2: FFT demagnetization field (NEW)
   *
   * Algorithm (implements the convolution theorem from the images):
   *   M̂(k) = FFT[M(r)]                   via cufftExecD2Z
   *   Ĥ(k) = D̂(k) · M̂(k)                pointwise complex multiply
   *   h(r)  = IFFT[Ĥ(k)] / (nx*ny)       via cufftExecZ2D + normalize
   *
   *   then:  ydot += LLG(m, h_dmag)       via demag_correction_kernel
   */
  if (udata->demag && DEMAG_STRENGTH > 0.0) {
    /* Zero the demag field buffer before accumulation */
    cudaMemsetAsync(udata->d_hdmag, 0,
                    (size_t)3 * udata->ncell * sizeof(sunrealtype), 0);

    /* Compute h_dmag = IFFT[ D̂ · FFT[M] ] — adds into d_hdmag */
    Demag_Apply(udata->demag,
                (const double*)ydata,
                (double*)udata->d_hdmag);

    /* Add LLG(m, h_dmag) correction to ydot */
    const int block1d = 256;
    const int grid1d  = (udata->ncell + block1d - 1) / block1d;
    demag_correction_kernel<<<grid1d, block1d>>>(
        ydata, udata->d_hdmag, ydotdata, udata->ncell);
  }

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
static void WriteFrame(FILE* fp, sunrealtype t,
                       int nx, int ny, int ng, int ncell, N_Vector y) {
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
  if (t <= SUN_RCONST(EARLY_SAVE_UNTIL))
    return (iout % EARLY_SAVE_EVERY) == 0;
  else
    return (iout % LATE_SAVE_EVERY) == 0;
}
#endif

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

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
  if (!fp) { fprintf(stderr, "Error opening output file.\n"); return 1; }
  setvbuf(fp, NULL, _IOFBF, 1 << 20);
#endif

  const double cx     = CIRCLE_CENTER_X_FRAC * (double)(ng - 1);
  const double cy     = CIRCLE_CENTER_Y_FRAC * (double)(ny - 1);
  const double radius = CIRCLE_RADIUS_FRAC_Y * (double)ny;

  udata.nx    = nx;
  udata.ny    = ny;
  udata.ng    = ng;
  udata.ncell = ncell;
  udata.neq   = neq;
  udata.pd    = NULL;
  udata.demag = NULL;
  udata.d_hdmag = NULL;

  CHECK_SUNDIALS(SUNContext_Create(SUN_COMM_NULL, &sunctx));

  /* 3x3 block Jacobi preconditioner */
  udata.pd = Precond_Create(ng, ny, ncell);
  if (!udata.pd) { fprintf(stderr, "Precond_Create failed\n"); return 1; }

  /* FFT Demagnetization field  <<< NEW
   *
   * Initializes cuFFT plans and precomputes D̂(k) (dipolar tensor in k-space).
   * Set DEMAG_STRENGTH=0.0 to disable (no overhead in f() either).
   */
  const double demag_str = (double)DEMAG_STRENGTH;
  if (demag_str > 0.0) {
    udata.demag = Demag_Init(ng, ny, demag_str);
    if (!udata.demag) {
      fprintf(stderr, "Demag_Init failed\n");
      Precond_Destroy(udata.pd);
      return 1;
    }
    /* Persistent device buffer for h_dmag (SoA, 3 components) */
    CHECK_CUDA(cudaMalloc((void**)&udata.d_hdmag,
                          (size_t)3 * ncell * sizeof(sunrealtype)));
  }

  y      = N_VNew_Cuda(neq, sunctx);
  abstol = N_VNew_Cuda(neq, sunctx);
  if (!y || !abstol) {
    fprintf(stderr, "Failed to allocate N_Vector_Cuda objects.\n");
    goto cleanup;
  }

  FusedNVec_Init(y);

  ydata       = N_VGetHostArrayPointer_Cuda(y);
  abstol_data = N_VGetHostArrayPointer_Cuda(abstol);
  if (!ydata || !abstol_data) {
    fprintf(stderr, "Failed to get host array pointers.\n");
    goto cleanup;
  }

  /* Initialize y (smooth Néel-like texture) and abstol */
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

        const double dx  = (double)i - cx;
        const double dy  = (double)j - cy;
        const double rho = sqrt(dx*dx + dy*dy);
        const double u   = (rho - radius) / width;
        const double s   = 0.5 * (1.0 - tanh(u));

        double mz0 = outer_mz + (core_mz - outer_mz) * s;
        if (mz0 >  1.0) mz0 =  1.0;
        if (mz0 < -1.0) mz0 = -1.0;

        double mperp = sqrt(fmax(0.0, 1.0 - mz0*mz0));
        double mx0, my0;
        if (rho > (double)TEXTURE_EPS) {
          mx0 = mperp * (dx / rho);
          my0 = mperp * (dy / rho);
        } else {
          mx0 = 0.0;  my0 = 0.0;
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
  if (!cvode_mem) { fprintf(stderr, "CVodeCreate failed.\n"); goto cleanup; }

  CHECK_SUNDIALS(CVodeInit(cvode_mem, f, T0, y));
  CHECK_SUNDIALS(CVodeSetUserData(cvode_mem, &udata));
  CHECK_SUNDIALS(CVodeSVtolerances(cvode_mem, RTOL, abstol));

  NLS = SUNNonlinSol_Newton(y, sunctx);
  if (!NLS) { fprintf(stderr, "SUNNonlinSol_Newton failed.\n"); goto cleanup; }
  CHECK_SUNDIALS(CVodeSetNonlinearSolver(cvode_mem, NLS));

  LS = SUNLinSol_SPGMR(y, SUN_PREC_LEFT, KRYLOV_DIM, sunctx);
  if (!LS) { fprintf(stderr, "SUNLinSol_SPGMR failed.\n"); goto cleanup; }
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

  printf("\n2D periodic smooth-texture LLG + FFT Demag (SoA, Tier1+3)\n\n");
  printf("nx=%d  ny=%d  ng=%d  ncell=%d  neq=%d\n", nx, ny, ng, ncell, neq);
  printf("periodic BC: x and y\n");
  printf("circle center=(%.2f,%.2f), radius=%.2f cells\n", cx, cy, radius);
  printf("DEMAG_STRENGTH = %.4f  (%s)\n",
         (double)DEMAG_STRENGTH,
         (demag_str > 0.0) ? "h_dmag = IFFT[D̂·M̂] via cuFFT" : "disabled");
  printf("T_TOTAL = %.2f   RTOL/ATOL = %.1e\n",
         (double)T_TOTAL, (double)RTOL_VAL);

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
    if (ShouldWriteFrame(iout + 1, t))
      WriteFrame(fp, t, nx, ny, ng, ncell, y);
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
  if (LS)        SUNLinSolFree(LS);
  if (NLS)       SUNNonlinSolFree(NLS);
  if (cvode_mem) CVodeFree(&cvode_mem);
  if (y)         N_VDestroy(y);
  if (abstol)    N_VDestroy(abstol);
  if (sunctx)    SUNContext_Free(&sunctx);
  Precond_Destroy(udata.pd);
  Demag_Destroy(udata.demag);              /* <<< FFT DEMAG cleanup */
  if (udata.d_hdmag) cudaFree(udata.d_hdmag);
  FusedNVec_FreePool();

#if ENABLE_OUTPUT
  if (fp) fclose(fp);
#endif

  return 0;
}
