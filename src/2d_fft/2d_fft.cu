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
#include <cufft.h> 
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

// ===================== Compile-time constants (放最前面) =====================
constexpr int NX = 128, NY = 128; // original picture size
constexpr int PX = 2 * NX; // linear convolution padding - width
constexpr int PY = 2 * NY; // linear convolution padding - height
constexpr int SIZE = PX * PY;

constexpr int BATCH_M = 3; // Mx, My, Mz
constexpr int BATCH_D = 9; // 9 core: Dxx..Dzz

// 你的 GPU 架构（与 nvcc -arch=sm_89 一致）
constexpr int CC  = 890; // Ada: sm_89
constexpr int EPT = 8; // element per thread
constexpr int FPB = 1; // #FFT per block

// constant memory
__constant__ float msk[3] = {0.0f, 0.0f, 1.0f};
__constant__ float nsk[3] = {1.0f, 0.0f, 0.0f};
__constant__ float chk = 1.0f;
__constant__ float che = 4.0f;
__constant__ float alpha = 0.2f; // 0.0f
__constant__ float chg = 1.0f;
__constant__ float cha = 0.0f; // 0.2
__constant__ float chb = 0.3f;

// transpose
#ifndef TILE_DIM
#define TILE_DIM 32
#endif
#ifndef BLOCK_ROWS
#define BLOCK_ROWS 8
#endif

using Complex = cufftComplex;
using namespace cufftdx;

// row is PX - FFT（forward / inverse）
using FFTxF = decltype(
  Size<PX>() + Type<fft_type::c2c>()
+ Direction<fft_direction::forward>()
+ Precision<float>() + Block()
+ ElementsPerThread<EPT>() + FFTsPerBlock<FPB>()
+ SM<CC>());

using FFTxI = decltype(
  Size<PX>() + Type<fft_type::c2c>()
+ Direction<fft_direction::inverse>()
+ Precision<float>() + Block()
+ ElementsPerThread<EPT>() + FFTsPerBlock<FPB>()
+ SM<CC>());

// row is PY - FFT (forward / inverse)
using FFTyF = decltype(
  Size<PY>() + Type<fft_type::c2c>()
+ Direction<fft_direction::forward>()
+ Precision<float>() + Block()
+ ElementsPerThread<EPT>() + FFTsPerBlock<FPB>()
+ SM<CC>());

using FFTyI = decltype(
  Size<PY>() + Type<fft_type::c2c>()
+ Direction<fft_direction::inverse>()
+ Precision<float>() + Block()
+ ElementsPerThread<EPT>() + FFTsPerBlock<FPB>()
+ SM<CC>());

/* user data structure for parallel*/
typedef struct {
  int nx, ny;
  int neq; // number of equations
  sunrealtype *d_h;
  sunrealtype *d_mh;

  // FFT resources
  Complex     *d_M;      // 3*SIZE, padded M (complex, real in .x)
  Complex     *d_H;      // 3*SIZE, padded H (complex)
  Complex     *d_tmp;    // SIZE, transpose/intermediate
  Complex     *d_Dhat;   // 9*SIZE, cached frequency-domain kernel
  bool         dhat_ready;
  sunrealtype *d_hfft;   // long-range field unpacked to y-layout (neq)
} UserData;

__global__ void complexMul(Complex *H, const Complex *A, const Complex *B, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float ar = A[idx].x, ai = A[idx].y;
    float br = B[idx].x, bi = B[idx].y;
    H[idx].x = ar * br - ai * bi;
    H[idx].y = ar * bi + ai * br;
  }
}
__global__ void complexMulAdd(Complex *H, const Complex *A, const Complex *B, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float ar = A[idx].x, ai = A[idx].y;
    float br = B[idx].x, bi = B[idx].y;
    H[idx].x += ar * br - ai * bi;
    H[idx].y += ar * bi + ai * br;
  }
}
__global__ void normalize(Complex *data, int N, float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    data[idx].x *= scale;
    data[idx].y *= scale;
  }
}

/* 
helper funcition
*/
inline __host__ __device__ int modp(int a, int p) {
  int r = a % p;
  return r < 0 ? r + p : r;
}

__global__ void pack3_to_padded_sun(const sunrealtype* __restrict__ y,
                                    Complex* __restrict__ M, // 3*SIZE
                                    int nx, int ny, int px, int py) {
  const int NCX = nx / 3;
  int i = blockIdx.x * blockDim.x + threadIdx.x; // cell x
  int j = blockIdx.y * blockDim.y + threadIdx.y; // cell y
  if (i >= px || j >= py) return;

  int pad_idx = i + j * px;
  if (i < NCX && j < ny) {
    int base = j * nx + i * 3;
    float mx = (float)y[base + 0];
    float my = (float)y[base + 1];
    float mz = (float)y[base + 2];
    M[0*SIZE + pad_idx].x = mx; M[0*SIZE + pad_idx].y = 0.f;
    M[1*SIZE + pad_idx].x = my; M[1*SIZE + pad_idx].y = 0.f;
    M[2*SIZE + pad_idx].x = mz; M[2*SIZE + pad_idx].y = 0.f;
  } else {
    M[0*SIZE + pad_idx].x = 0.f; M[0*SIZE + pad_idx].y = 0.f;
    M[1*SIZE + pad_idx].x = 0.f; M[1*SIZE + pad_idx].y = 0.f;
    M[2*SIZE + pad_idx].x = 0.f; M[2*SIZE + pad_idx].y = 0.f;
  }
}

__global__ void scatter3_from_padded_sun(const Complex* __restrict__ H, // 3*SIZE
                                         sunrealtype* __restrict__ hfft, // neq
                                         int nx, int ny, int px) {
  const int NCX = nx / 3;
  int i = blockIdx.x * blockDim.x + threadIdx.x; // cell x
  int j = blockIdx.y * blockDim.y + threadIdx.y; // cell y
  if (i >= NCX || j >= ny) return;

  int pad_idx = i + j * px;
  int base = j * nx + i * 3;
  hfft[base + 0] = (sunrealtype)H[0*SIZE + pad_idx].x;
  hfft[base + 1] = (sunrealtype)H[1*SIZE + pad_idx].x;
  hfft[base + 2] = (sunrealtype)H[2*SIZE + pad_idx].x;
}

// transpose, no bank conflict
__global__ void transpose_tiles(Complex* __restrict__ out,
                                const Complex* __restrict__ in,
                                int pitch_x, int pitch_y, int batches) {
  __shared__ Complex tile[TILE_DIM][TILE_DIM + 1];
  int b = blockIdx.z;
  size_t plane_off = size_t(b) * size_t(pitch_x) * size_t(pitch_y);
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  #pragma unroll
  for (int r = 0; r < TILE_DIM; r += BLOCK_ROWS) {
    int yy = y + r;
    if (x < pitch_x && yy < pitch_y)
      tile[threadIdx.y + r][threadIdx.x] = in[plane_off + size_t(yy) * pitch_x + x];
  }
  __syncthreads();

  int xt = blockIdx.y * TILE_DIM + threadIdx.x;
  int yt = blockIdx.x * TILE_DIM + threadIdx.y;

  #pragma unroll
  for (int r = 0; r < TILE_DIM; r += BLOCK_ROWS) {
    int yyt = yt + r;
    if (xt < pitch_y && yyt < pitch_x)
      out[plane_off + size_t(yyt) * pitch_y + xt] = tile[threadIdx.x][threadIdx.y + r];
  }
}

// row FFT kernel
template <class FFTX>
__global__ void fft_rows_c2c(Complex* __restrict__ buf, int pitch_x, int pitch_y, int batches){
  using FFT = FFTX;
  extern __shared__ __align__(16) unsigned char smem[];

  const int global_fft_id = blockIdx.x;
  const int row   = global_fft_id % pitch_y;
  const int batch = (global_fft_id / pitch_y) % batches;

  const size_t plane_off = size_t(batch) * size_t(pitch_x) * size_t(pitch_y);
  const size_t base = plane_off + size_t(row) * size_t(pitch_x);

  using VT = typename FFT::value_type;
  VT thread_data[FFT::storage_size];

  unsigned t = threadIdx.x;
  #pragma unroll
  for (int i = 0; i < FFT::elements_per_thread; i++) {
    unsigned k = t + i * FFT::stride;
    if (k < cufftdx::size_of<FFT>::value) {
      const Complex* g = &buf[base + k];
      thread_data[i] = VT{ g->x, g->y };
    }
  }

  FFT().execute(thread_data, smem);

  t = threadIdx.x;
  #pragma unroll
  for (int i = 0; i < FFT::elements_per_thread; i++) {
    unsigned k = t + i * FFT::stride;
    if (k < cufftdx::size_of<FFT>::value) {
      Complex* g = &buf[base + k];
      const VT& v = thread_data[i];
      g->x = static_cast<float>(v.real());
      g->y = static_cast<float>(v.imag());
    }
  }
}

// 2d FFT, IFFT
template<class FFT_ROW, class FFT_COL>
inline void fft2_forward_ud(Complex* buf, int batches, UserData* u) {
  dim3 grid_rows(PY * batches);
  dim3 block_rows  = FFT_ROW::block_dim;
  size_t shmem = FFT_ROW::shared_memory_size;
  fft_rows_c2c<FFT_ROW><<<grid_rows, block_rows, shmem>>>(buf, PX, PY, batches);

  dim3 gridT( (PX+TILE_DIM-1)/TILE_DIM, (PY+TILE_DIM-1)/TILE_DIM, batches );
  dim3 blockT(TILE_DIM, BLOCK_ROWS, 1);
  transpose_tiles<<<gridT, blockT>>>(u->d_tmp, buf, PX, PY, batches);

  dim3 grid_rows2(PX * batches);
  dim3 block_rows2 = FFT_COL::block_dim;
  size_t shmem2 = FFT_COL::shared_memory_size;
  fft_rows_c2c<FFT_COL><<<grid_rows2, block_rows2, shmem2>>>(u->d_tmp, PY, PX, batches);

  dim3 gridT2( (PY+TILE_DIM-1)/TILE_DIM, (PX+TILE_DIM-1)/TILE_DIM, batches );
  transpose_tiles<<<gridT2, blockT>>>(buf, u->d_tmp, PY, PX, batches);
}

template<class FFT_ROW, class FFT_COL>
inline void fft2_inverse_ud(Complex* buf, int batches, UserData* u) {
  dim3 grid_rows(PY * batches);
  dim3 block_rows  = FFT_ROW::block_dim;
  size_t shmem = FFT_ROW::shared_memory_size;
  fft_rows_c2c<FFT_ROW><<<grid_rows, block_rows, shmem>>>(buf, PX, PY, batches);

  dim3 gridT( (PX+TILE_DIM-1)/TILE_DIM, (PY+TILE_DIM-1)/TILE_DIM, batches );
  dim3 blockT(TILE_DIM, BLOCK_ROWS, 1);
  transpose_tiles<<<gridT, blockT>>>(u->d_tmp, buf, PX, PY, batches);

  dim3 grid_rows2(PX * batches);
  dim3 block_rows2 = FFT_COL::block_dim;
  size_t shmem2 = FFT_COL::shared_memory_size;
  fft_rows_c2c<FFT_COL><<<grid_rows2, block_rows2, shmem2>>>(u->d_tmp, PY, PX, batches);

  dim3 gridT2( (PY+TILE_DIM-1)/TILE_DIM, (PX+TILE_DIM-1)/TILE_DIM, batches );
  transpose_tiles<<<gridT2, blockT>>>(buf, u->d_tmp, PY, PX, batches);
}

static void buildD_host(Complex* hD, int nx, int ny, int px, int py) {
  for (int c = 0; c < 9 * SIZE; c++) { hD[c].x = 0.f; hD[c].y = 0.f; }
  for (int uy = -ny + 1; uy <= ny - 1; ++uy) {
    for (int ux = -nx + 1; ux <= nx - 1; ++ux) {
      int x = modp(ux, px);
      int y = modp(uy, py);
      int idx = x + y * px;
      float Dxx = (ux == 0 && uy == 0) ? 1.f : 0.f;
      hD[0*SIZE + idx].x = Dxx;
    }
  }
}

/*
 *-------------------------------
 * Functions called by the solver
 *-------------------------------
 */

/* Right hand side function evaluation kernel. */
__global__ static void f_kernel(const sunrealtype *y, sunrealtype *yd,
                                sunrealtype *h, sunrealtype *mh, 
                                const sunrealtype *hfft,
                                int nx, int ny) {
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

    float h_local = che * (y[mxq] + y[mxp] + y[myq] + y[myp])
                  + msk[imsk] * (chk * y[mz] + cha)
                  + chb * nsk[imsk] * (y[mxq] + y[mxp]);

    h[tid] = h_local + hfft[tid];
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

static inline dim3 grid2D(int X, int Y, int bx=32, int by=16) {
  return dim3((X + bx - 1) / bx, (Y + by - 1) / by);
}


static void update_longrange(UserData* U, const sunrealtype* ydev) {
  // 1) package y to padded M
  dim3 B(32,16), G = grid2D(PX, PY, B.x, B.y);
  pack3_to_padded_sun<<<G, B>>>(ydev, U->d_M, U->nx, U->ny, PX, PY);

  // 2) FFT2(M)
  fft2_forward_ud<FFTxF, FFTyF>(U->d_M, /*batches=*/3, U);

  // 3) Hhat = Dhat * Mhat（3*3）
  Complex* Dxx=U->d_Dhat+0*SIZE; Complex* Dxy=U->d_Dhat+1*SIZE; Complex* Dxz=U->d_Dhat+2*SIZE;
  Complex* Dyx=U->d_Dhat+3*SIZE; Complex* Dyy=U->d_Dhat+4*SIZE; Complex* Dyz=U->d_Dhat+5*SIZE;
  Complex* Dzx=U->d_Dhat+6*SIZE; Complex* Dzy=U->d_Dhat+7*SIZE; Complex* Dzz=U->d_Dhat+8*SIZE;

  Complex* Mx=U->d_M+0*SIZE; Complex* My=U->d_M+1*SIZE; Complex* Mz=U->d_M+2*SIZE;
  Complex* Hx=U->d_H+0*SIZE; Complex* Hy=U->d_H+1*SIZE; Complex* Hz=U->d_H+2*SIZE;

  int thr=256, blk=(SIZE+thr-1)/thr;
  complexMul   <<<blk,thr>>>(Hx, Dxx, Mx, SIZE);
  complexMulAdd<<<blk,thr>>>(Hx, Dxy, My, SIZE);
  complexMulAdd<<<blk,thr>>>(Hx, Dxz, Mz, SIZE);

  complexMul   <<<blk,thr>>>(Hy, Dyx, Mx, SIZE);
  complexMulAdd<<<blk,thr>>>(Hy, Dyy, My, SIZE);
  complexMulAdd<<<blk,thr>>>(Hy, Dyz, Mz, SIZE);

  complexMul   <<<blk,thr>>>(Hz, Dzx, Mx, SIZE);
  complexMulAdd<<<blk,thr>>>(Hz, Dzy, My, SIZE);
  complexMulAdd<<<blk,thr>>>(Hz, Dzz, Mz, SIZE);
  cudaGetLastError();

  // 4) iFFT2(H) + normalized
  fft2_inverse_ud<FFTxI, FFTyI>(U->d_H, /*batches=*/3, U);
  normalize<<<(3*SIZE+thr-1)/thr,thr>>>(U->d_H, 3*SIZE, 1.0f/float(SIZE));

  // 5) unpack to y - hfft
  scatter3_from_padded_sun<<<grid2D(U->nx/3, U->ny, B.x, B.y), B>>>(U->d_H, U->d_hfft, U->nx, U->ny, PX);
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

  if (!udata->dhat_ready) {
    fprintf(stderr, "ERROR: Dhat not ready. Build it in main() before CVode.\n");
    return -1;
  }
  update_longrange(udata, ydata);

  int nx = udata->nx, ny = udata->ny;

  int dimx = 30;
  int dimy = 32;
  dim3 block(dimx, dimy);
  int blocks_x = (nx + block.x - 1) / block.x;
  int blocks_y = (ny + block.y - 1) / block.y;
  dim3 grid(blocks_x, blocks_y);

  // printf("grid: %d %d\n",blocks_x, blocks_y);
  // printf("block: %d %d\n",block.x, block.y);
  f_kernel<<<grid, block>>>(ydata, ydotdata, udata->d_h, udata->d_mh,
                          udata->d_hfft,  // <<< 新增
                          nx, ny);
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
  int nx = 3 * NX;  // 每个 cell 3 分量 → y 宽度是 3*NX
  int ny = NY;
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

  cudaMalloc(&udata.d_M,   BATCH_M * SIZE * sizeof(Complex));
  cudaMalloc(&udata.d_H,   BATCH_M * SIZE * sizeof(Complex));
  cudaMalloc(&udata.d_tmp, SIZE * sizeof(Complex));
  cudaMalloc(&udata.d_Dhat,BATCH_D * SIZE * sizeof(Complex));
  cudaMalloc(&udata.d_hfft, udata.neq * sizeof(sunrealtype));

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

  {
    Complex *h_D = (Complex*)malloc(BATCH_D * SIZE * sizeof(Complex));
    if (!h_D) { fprintf(stderr, "Host alloc failed for h_D\n"); return 1; }
    buildD_host(h_D, NX, NY, PX, PY);
    Complex *d_Dspatial = nullptr;
    cudaMalloc(&d_Dspatial, BATCH_D * SIZE * sizeof(Complex));
    cudaMemcpy(d_Dspatial, h_D, BATCH_D * SIZE * sizeof(Complex), cudaMemcpyHostToDevice);

    // NEW: prepare frequency-domain kernel once
    cudaMemcpy(udata.d_Dhat, d_Dspatial, BATCH_D * SIZE * sizeof(Complex),
              cudaMemcpyDeviceToDevice);                                  // NEW
    fft2_forward_ud<FFTxF, FFTyF>(udata.d_Dhat, /*batches=*/BATCH_D, &udata); // NEW
    udata.dhat_ready = true;                                               // NEW

    cudaFree(d_Dspatial);
    free(h_D);
  }

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
  printf("\nCoupled local (4-neigh) + long-range (FFT) system\n");
  printf("cells = %d x %d ; y-grid = %d x %d ; neq = %d\n", NX, NY, nx, ny, nx*ny);

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
  cudaFree(udata.d_M);
  cudaFree(udata.d_H);
  cudaFree(udata.d_tmp);
  cudaFree(udata.d_Dhat);
  cudaFree(udata.d_hfft);
  N_VDestroy(y);
  N_VDestroy(abstol);
  CVodeFree(&cvode_mem);
  SUNLinSolFree(LS);
  SUNNonlinSolFree(NLS);
  SUNContext_Free(&sunctx);
  fclose(fp);

  return 0;
}
