/*
 * fft_demag.cu  —  FFT-based demagnetization field for 2D periodic LLG
 *
 * Physics / Math (from the handwritten derivation in the images):
 * ----------------------------------------------------------------
 *  Space domain:
 *    h_dmag(i,j) = Σ_{m,n}  D(i-m, j-n) · M(m,n)          [convolution]
 *
 *  Convolution theorem (DFT both sides):
 *    Σ_{i,j} h(i,j) e^{-i(kx·i + ky·j)} = ĥ(kx, ky)
 *
 *  Factor the double sum:
 *    = Σ_{m,n} M(m,n) Σ_{i,j} D(i-m, j-n) e^{-i(kx·i + ky·j)}
 *    = Σ_{m,n} M(m,n) e^{-i(kx·m + ky·n)} · D̂(kx, ky)
 *    = M̂(kx, ky) · D̂(kx, ky)
 *
 *  Therefore:
 *    h_dmag(i,j) = IFFT[ D̂(kx,ky) · M̂(kx,ky) ]
 *
 * Algorithm (one call to Demag_Apply per timestep):
 *   1.  R2C FFT of mx, my, mz separately  →  M̂x, M̂y, M̂z
 *   2.  Pointwise multiply with precomputed D̂ (3×3 tensor in k-space):
 *         Ĥx = D̂xx·M̂x + D̂xy·M̂y + D̂xz·M̂z
 *         Ĥy = D̂yx·M̂x + D̂yy·M̂y + D̂yz·M̂z
 *         Ĥz = D̂zx·M̂x + D̂zy·M̂y + D̂zz·M̂z
 *   3.  C2R IFFT of Ĥx, Ĥy, Ĥz  →  h_dmag_x, h_dmag_y, h_dmag_z
 *   4.  Normalize by 1/(nx*ny)
 *
 * Demag tensor D (2D film, periodic boundary, dipole-dipole):
 *   For a 2D thin-film periodic system the real-space dipolar kernel is:
 *     D_αβ(r) = (3 r_α r_β / r^5) - δ_αβ / r^3
 *   We compute D̂_αβ(k) analytically in k-space for the periodic case.
 *   The standard result for a 2D periodic dipolar system is:
 *     D̂_xx(kx,ky) = 2π * kx^2 / k   (k = sqrt(kx^2+ky^2))
 *     D̂_yy(kx,ky) = 2π * ky^2 / k
 *     D̂_zz(kx,ky) = -2π * k         (out-of-plane)
 *     D̂_xy(kx,ky) = 2π * kx*ky / k
 *     D̂_xz = D̂_yz = 0              (in-plane k, uniform thickness)
 *   At k=0 (uniform mode): D̂_xx=D̂_yy=0, D̂_zz depends on sample shape.
 *   For a laterally infinite thin film: D̂_zz(0)=0, demagnetization fully
 *   accounted for by the c_chk easy-plane term in the RHS.
 *
 * Integration into 2d_p.cu:
 *   Add  Demag_Apply(udata->demag, y, ydot_or_hfield)  inside f()
 *   after the exchange stencil, adding h_dmag_x/y/z to the effective field.
 *
 * Memory layout:
 *   SoA: [mx_0..mx_{N-1}][my_0..my_{N-1}][mz_0..mz_{N-1}]  (same as 2d_p)
 *   FFT buffers: separate padded real arrays per component (nx * ny doubles)
 *   Spectral:    complex arrays of size nx * (ny/2+1) per component
 *
 * cuFFT types used (double precision, matching sunrealtype = double):
 *   cufftPlan2d(ny, nx, CUFFT_D2Z)   forward R→C
 *   cufftPlan2d(ny, nx, CUFFT_Z2D)   inverse C→R
 *   (Note: cuFFT nx is the slowest / row dimension = ny in our notation)
 *
 * Performance notes:
 *   - D̂ is precomputed once in Demag_Init, stored on device (6 arrays,
 *     each nx*(ny/2+1) complex doubles = 6 * nx*(ny/2+1)*16 bytes).
 *   - Forward/inverse FFTs: 6 plans × 1 call per f() evaluation.
 *   - Pointwise multiply: 1 kernel, reads 3 M̂ + 6 D̂, writes 3 Ĥ.
 *   - For nx=1000, ny=1280 on sm_89: expect ~2ms per f() call for FFTs.
 *
 * Compile:  nvcc -O3 -arch=sm_89 fft_demag.cu -lcufft -o test_demag
 */

#include "demag_fft.h"

#include <cufft.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <nvector/nvector_cuda.h>

/* -----------------------------------------------------------------------
 * Internal error macros
 * --------------------------------------------------------------------- */
#define CHECK_CUDA_FD(call) do {                                            \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "[fft_demag] CUDA error %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        return NULL;                                                        \
    }                                                                      \
} while(0)

#define CHECK_CUFFT(call) do {                                              \
    cufftResult _r = (call);                                                \
    if (_r != CUFFT_SUCCESS) {                                              \
        fprintf(stderr, "[fft_demag] cuFFT error %s:%d: code=%d\n",        \
                __FILE__, __LINE__, (int)_r);                               \
        return NULL;                                                        \
    }                                                                      \
} while(0)

#define CHECK_CUDA_VOID(call) do {                                          \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "[fft_demag] CUDA error %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
    }                                                                      \
} while(0)

/* -----------------------------------------------------------------------
 * DemagData struct (opaque to caller — defined here, declared in header)
 * --------------------------------------------------------------------- */
struct DemagData {
    int nx;              /* number of columns (fast index in LLG) */
    int ny;              /* number of rows    (slow index in LLG) */
    int ncell;           /* nx * ny                               */
    int nk;              /* nx * (ny/2 + 1)   = complex spectrum size */

    /* cuFFT plans: forward (real → complex) and inverse (complex → real) */
    cufftHandle planR2C;   /* D2Z: ny × nx real  →  ny × (nx/2+1) complex */
    cufftHandle planC2R;   /* Z2D: ny × (nx/2+1) complex → ny × nx real   */

    /* Device buffers: real scratch (one component at a time) */
    double *d_real;        /* nx * ny doubles  */

    /* Device buffers: complex spectra for M and H (3 components each) */
    cufftDoubleComplex *d_Mhat[3];   /* M̂x, M̂y, M̂z */
    cufftDoubleComplex *d_Hhat[3];   /* Ĥx, Ĥy, Ĥz  */

    /* Precomputed demag tensor in k-space (6 independent components,
     * symmetric: Dxx, Dyy, Dzz, Dxy, Dxz, Dyz) */
    cufftDoubleComplex *d_Dhat[6];
    /* Index mapping: 0=xx, 1=yy, 2=zz, 3=xy, 4=xz, 5=yz */

    double scale;     /* normalization: 1.0/(nx*ny) after IFFT */
    double demag_strength; /* prefactor for demag field (default 1.0) */
};

/* -----------------------------------------------------------------------
 * Kernel: precompute demag tensor D̂ in k-space
 *
 * For a 2D periodic dipolar thin film the analytical k-space demag tensor
 * (Newell et al., or equivalent) is:
 *
 *   kx_phys = 2π * kx_idx / nx    (wavenumber in x, in units of 1/cell)
 *   ky_phys = 2π * ky_idx / ny
 *   k = sqrt(kx^2 + ky^2)
 *
 *   D̂_xx = 2π * kx^2 / k        D̂_yy = 2π * ky^2 / k
 *   D̂_zz = -2π * k              D̂_xy = 2π * kx*ky / k
 *   D̂_xz = D̂_yz = 0
 *
 * At k=0: D̂_xx = D̂_yy = D̂_zz = 0 (handled by shape anisotropy term c_chk)
 *
 * The result is purely real (no imaginary part) because the dipolar kernel
 * is symmetric about the origin.  We store as cufftDoubleComplex with im=0.
 *
 * Grid: one thread per k-space point (ky_idx, kx_idx)
 *   kx_idx runs 0..nx/2   (R2C output, only half-spectrum)
 *   ky_idx runs 0..ny-1
 * --------------------------------------------------------------------- */
__global__ static void demag_build_Dhat_kernel(
    cufftDoubleComplex *Dxx,
    cufftDoubleComplex *Dyy,
    cufftDoubleComplex *Dzz,
    cufftDoubleComplex *Dxy,
    cufftDoubleComplex *Dxz,
    cufftDoubleComplex *Dyz,
    int nx, int ny,   /* physical grid: nx cols, ny rows */
    int nkx)          /* number of kx points = nx/2+1 */
{
    const int ikx = blockIdx.x * blockDim.x + threadIdx.x;  /* 0..nkx-1 */
    const int iky = blockIdx.y * blockDim.y + threadIdx.y;  /* 0..ny-1  */

    if (ikx >= nkx || iky >= ny) return;

    const int idx = iky * nkx + ikx;   /* linear index in half-spectrum */

    /* Physical wavenumbers in units of [1/cell spacing]
     * ky: use negative frequencies for iky > ny/2 */
    double kx = (2.0 * M_PI / nx) * ikx;
    double ky_raw = (iky <= ny/2) ? (double)iky : (double)(iky - ny);
    double ky = (2.0 * M_PI / ny) * ky_raw;

    double k2 = kx*kx + ky*ky;
    double k  = sqrt(k2);

    double Dxx_val, Dyy_val, Dzz_val, Dxy_val;

    if (k < 1.0e-14) {
        /* k=0: uniform mode — set to zero (shape anisotropy handled by c_chk) */
        Dxx_val = 0.0;
        Dyy_val = 0.0;
        Dzz_val = 0.0;
        Dxy_val = 0.0;
    } else {
        /* Standard 2D dipolar demag tensor (thin film, periodic) */
        double inv_k = 1.0 / k;
        Dxx_val =  2.0 * M_PI * kx * kx * inv_k;   /* > 0 */
        Dyy_val =  2.0 * M_PI * ky * ky * inv_k;   /* > 0 */
        Dzz_val = -2.0 * M_PI * k;                  /* < 0 */
        Dxy_val =  2.0 * M_PI * kx * ky * inv_k;
    }

    /* Store as purely real complex numbers (imaginary part = 0) */
    Dxx[idx].x = Dxx_val;  Dxx[idx].y = 0.0;
    Dyy[idx].x = Dyy_val;  Dyy[idx].y = 0.0;
    Dzz[idx].x = Dzz_val;  Dzz[idx].y = 0.0;
    Dxy[idx].x = Dxy_val;  Dxy[idx].y = 0.0;
    Dxz[idx].x = 0.0;      Dxz[idx].y = 0.0;   /* in-plane k → zero */
    Dyz[idx].x = 0.0;      Dyz[idx].y = 0.0;
}

/* -----------------------------------------------------------------------
 * Kernel: copy one SoA component (stride=ncell) to a packed real buffer
 *
 * SoA layout:  y[comp*ncell + cell]   (cell = j*nx + i)
 * Packed:      buf[j*nx + i]
 *
 * These are the same layout — it's just a strided gather.
 * For mx: comp=0,  for my: comp=1,  for mz: comp=2.
 * --------------------------------------------------------------------- */
__global__ static void demag_gather_component(
    const double* __restrict__ y_soa,    /* full SoA state vector */
    double*       __restrict__ buf,      /* nx*ny real scratch    */
    int comp, int ncell)
{
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= ncell) return;
    buf[cell] = y_soa[comp * ncell + cell];
}

/* -----------------------------------------------------------------------
 * Kernel: scatter one real buffer back to SoA hdmag accumulator
 *
 * Adds (not overwrites) so we can call for x, y, z sequentially.
 * --------------------------------------------------------------------- */
__global__ static void demag_scatter_add(
    const double* __restrict__ buf,
    double*       __restrict__ h_soa,   /* hdmag accumulator, SoA */
    int comp, int ncell, double scale)
{
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= ncell) return;
    h_soa[comp * ncell + cell] += scale * buf[cell];
}

/* -----------------------------------------------------------------------
 * Kernel: pointwise multiply  Ĥ_α = Σ_β D̂_αβ · M̂_β
 *
 * For each k-space point:
 *   Ĥx = Dxx*Mx + Dxy*My + Dxz*Mz
 *   Ĥy = Dxy*Mx + Dyy*My + Dyz*Mz     (Dyx=Dxy by symmetry)
 *   Ĥz = Dxz*Mx + Dyz*My + Dzz*Mz     (Dzx=Dxz, Dzy=Dyz)
 *
 * Since D̂ is purely real:  (D̂·M̂) = D̂.re * M̂
 * --------------------------------------------------------------------- */
__global__ static void demag_multiply_kernel(
    const cufftDoubleComplex* __restrict__ Mx,
    const cufftDoubleComplex* __restrict__ My,
    const cufftDoubleComplex* __restrict__ Mz,
    const cufftDoubleComplex* __restrict__ Dxx,
    const cufftDoubleComplex* __restrict__ Dyy,
    const cufftDoubleComplex* __restrict__ Dzz,
    const cufftDoubleComplex* __restrict__ Dxy,
    const cufftDoubleComplex* __restrict__ Dxz,
    const cufftDoubleComplex* __restrict__ Dyz,
    cufftDoubleComplex* __restrict__ Hx,
    cufftDoubleComplex* __restrict__ Hy,
    cufftDoubleComplex* __restrict__ Hz,
    int nk)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nk) return;

    /* Load M̂ components */
    const double Mxr = Mx[k].x,  Mxi = Mx[k].y;
    const double Myr = My[k].x,  Myi = My[k].y;
    const double Mzr = Mz[k].x,  Mzi = Mz[k].y;

    /* Load D̂ (purely real — .y should be 0) */
    const double dxx = Dxx[k].x;
    const double dyy = Dyy[k].x;
    const double dzz = Dzz[k].x;
    const double dxy = Dxy[k].x;
    const double dxz = Dxz[k].x;
    const double dyz = Dyz[k].x;

    /* Ĥα = Σ_β D̂_αβ · M̂_β  (complex multiplication with real D̂) */
    Hx[k].x = dxx*Mxr + dxy*Myr + dxz*Mzr;
    Hx[k].y = dxx*Mxi + dxy*Myi + dxz*Mzi;

    Hy[k].x = dxy*Mxr + dyy*Myr + dyz*Mzr;
    Hy[k].y = dxy*Mxi + dyy*Myi + dyz*Mzi;

    Hz[k].x = dxz*Mxr + dyz*Myr + dzz*Mzr;
    Hz[k].y = dxz*Mxi + dyz*Myi + dzz*Mzi;
}

/* -----------------------------------------------------------------------
 * Public API: Demag_Init
 * --------------------------------------------------------------------- */
DemagData* Demag_Init(int nx, int ny, double demag_strength)
{
    DemagData *d = (DemagData*)calloc(1, sizeof(DemagData));
    if (!d) { fprintf(stderr, "[fft_demag] calloc failed\n"); return NULL; }

    d->nx = nx;
    d->ny = ny;
    d->ncell = nx * ny;
    d->nk    = ny * (nx/2 + 1);      /* half-spectrum size (R2C output) */
    d->scale = 1.0 / (double)(nx * ny);
    d->demag_strength = demag_strength;

    /* ---- cuFFT plans ----
     * cuFFT convention for 2D:
     *   cufftPlan2d(&plan, n0, n1, type)
     *   n0 = slowest (row) dimension = ny in our notation
     *   n1 = fastest (col) dimension = nx in our notation
     *   R2C output: n0 × (n1/2+1) complex
     * ----------------------------------------------------------------- */
    cufftResult r;
    r = cufftPlan2d(&d->planR2C, ny, nx, CUFFT_D2Z);
    if (r != CUFFT_SUCCESS) {
        fprintf(stderr, "[fft_demag] cufftPlan2d D2Z failed: %d\n", (int)r);
        free(d); return NULL;
    }
    r = cufftPlan2d(&d->planC2R, ny, nx, CUFFT_Z2D);
    if (r != CUFFT_SUCCESS) {
        fprintf(stderr, "[fft_demag] cufftPlan2d Z2D failed: %d\n", (int)r);
        cufftDestroy(d->planR2C); free(d); return NULL;
    }

    /* ---- Allocate real scratch (one component) ---- */
    CHECK_CUDA_FD(cudaMalloc((void**)&d->d_real,
                             (size_t)d->ncell * sizeof(double)));

    /* ---- Allocate M̂ and Ĥ complex buffers (3 each) ---- */
    for (int c = 0; c < 3; c++) {
        CHECK_CUDA_FD(cudaMalloc((void**)&d->d_Mhat[c],
                                 (size_t)d->nk * sizeof(cufftDoubleComplex)));
        CHECK_CUDA_FD(cudaMalloc((void**)&d->d_Hhat[c],
                                 (size_t)d->nk * sizeof(cufftDoubleComplex)));
    }

    /* ---- Allocate D̂ tensor (6 components) ---- */
    for (int c = 0; c < 6; c++) {
        CHECK_CUDA_FD(cudaMalloc((void**)&d->d_Dhat[c],
                                 (size_t)d->nk * sizeof(cufftDoubleComplex)));
    }

    /* ---- Precompute D̂ on device ---- */
    {
        const int nkx = nx/2 + 1;
        dim3 block(32, 8);
        dim3 grid((nkx + block.x - 1) / block.x,
                  (ny  + block.y - 1) / block.y);

        demag_build_Dhat_kernel<<<grid, block>>>(
            d->d_Dhat[0], d->d_Dhat[1], d->d_Dhat[2],
            d->d_Dhat[3], d->d_Dhat[4], d->d_Dhat[5],
            nx, ny, nkx);

        cudaError_t ce = cudaPeekAtLastError();
        if (ce != cudaSuccess) {
            fprintf(stderr, "[fft_demag] demag_build_Dhat_kernel failed: %s\n",
                    cudaGetErrorString(ce));
            Demag_Destroy(d);
            return NULL;
        }
        cudaDeviceSynchronize();
    }

    /* ---- Report memory usage ---- */
    size_t mem_Dhat = 6 * (size_t)d->nk * sizeof(cufftDoubleComplex);
    size_t mem_MH   = 6 * (size_t)d->nk * sizeof(cufftDoubleComplex);
    size_t mem_real = (size_t)d->ncell * sizeof(double);
    size_t mem_total = mem_Dhat + mem_MH + mem_real;

    printf("[Demag FFT] Initialized: nx=%d ny=%d ncell=%d nk=%d\n",
           nx, ny, d->ncell, d->nk);
    printf("[Demag FFT] Memory: Dhat=%.1f MB, MH_bufs=%.1f MB, real=%.1f MB"
           " | total=%.1f MB\n",
           mem_Dhat/(1.0e6), mem_MH/(1.0e6),
           mem_real/(1.0e6), mem_total/(1.0e6));
    printf("[Demag FFT] demag_strength = %.4f\n", demag_strength);
    printf("[Demag FFT] Algorithm: h_dmag = IFFT[ D̂(k) · M̂(k) ]\n");

    return d;
}

/* -----------------------------------------------------------------------
 * Public API: Demag_Destroy
 * --------------------------------------------------------------------- */
void Demag_Destroy(DemagData *d)
{
    if (!d) return;

    cufftDestroy(d->planR2C);
    cufftDestroy(d->planC2R);

    if (d->d_real) cudaFree(d->d_real);

    for (int c = 0; c < 3; c++) {
        if (d->d_Mhat[c]) cudaFree(d->d_Mhat[c]);
        if (d->d_Hhat[c]) cudaFree(d->d_Hhat[c]);
    }
    for (int c = 0; c < 6; c++) {
        if (d->d_Dhat[c]) cudaFree(d->d_Dhat[c]);
    }

    free(d);
}

/* -----------------------------------------------------------------------
 * Public API: Demag_Apply
 *
 * Computes h_dmag = IFFT[ D̂ · FFT[m] ] and ADDS it to h_out (SoA).
 *
 * Arguments:
 *   d        : DemagData handle from Demag_Init
 *   y_dev    : device pointer to SoA state [mx...my...mz...], size 3*ncell
 *   h_out    : device pointer to SoA field  [hx...hy...hz...], size 3*ncell
 *              (must be pre-zeroed or contain the exchange field to add to)
 *
 * Note: This function ADDS to h_out.  Zero it first if needed.
 * --------------------------------------------------------------------- */
void Demag_Apply(DemagData *d,
                 const double *y_dev,   /* device, SoA magnetization */
                 double       *h_out)   /* device, SoA field output  */
{
    const int ncell = d->ncell;
    const int nk    = d->nk;

    /* ---- Step 1: FFT each component of M ----
     *
     * Gather m_α from SoA stride → packed real buffer → D2Z FFT → M̂_α
     * (α = x, y, z = 0, 1, 2)
     * ----------------------------------------------------------------- */
    {
        const int block = 256;
        const int grid  = (ncell + block - 1) / block;

        for (int comp = 0; comp < 3; comp++) {
            /* Gather: d_real[cell] = y_dev[comp*ncell + cell] */
            demag_gather_component<<<grid, block>>>(
                y_dev, d->d_real, comp, ncell);

            /* Forward FFT: d_real (ny×nx real) → d_Mhat[comp] (ny×(nx/2+1) complex) */
            cufftResult r = cufftExecD2Z(d->planR2C,
                                         (cufftDoubleReal*)d->d_real,
                                         d->d_Mhat[comp]);
            if (r != CUFFT_SUCCESS) {
                fprintf(stderr, "[fft_demag] cufftExecD2Z comp=%d failed: %d\n",
                        comp, (int)r);
                return;
            }
        }
    }

    /* ---- Step 2: Pointwise tensor multiply in k-space ----
     *
     * Ĥ_α(k) = Σ_β D̂_αβ(k) · M̂_β(k)
     * ----------------------------------------------------------------- */
    {
        const int block = 256;
        const int grid  = (nk + block - 1) / block;

        demag_multiply_kernel<<<grid, block>>>(
            d->d_Mhat[0], d->d_Mhat[1], d->d_Mhat[2],
            d->d_Dhat[0], d->d_Dhat[1], d->d_Dhat[2],   /* xx, yy, zz */
            d->d_Dhat[3], d->d_Dhat[4], d->d_Dhat[5],   /* xy, xz, yz */
            d->d_Hhat[0], d->d_Hhat[1], d->d_Hhat[2],
            nk);
    }

    /* ---- Step 3: IFFT each component of Ĥ ----
     *
     * d_Hhat[comp] (complex) → d_real (real) → add scaled to h_out SoA
     * cuFFT C2R is unnormalized → divide by N = nx*ny
     * ----------------------------------------------------------------- */
    {
        const int block = 256;
        const int grid  = (ncell + block - 1) / block;
        const double s  = d->scale * d->demag_strength;

        for (int comp = 0; comp < 3; comp++) {
            /* Inverse FFT: d_Hhat[comp] → d_real */
            cufftResult r = cufftExecZ2D(d->planC2R,
                                         d->d_Hhat[comp],
                                         (cufftDoubleReal*)d->d_real);
            if (r != CUFFT_SUCCESS) {
                fprintf(stderr, "[fft_demag] cufftExecZ2D comp=%d failed: %d\n",
                        comp, (int)r);
                return;
            }

            /* Scatter-add: h_out[comp*ncell + cell] += s * d_real[cell] */
            demag_scatter_add<<<grid, block>>>(
                d->d_real, h_out, comp, ncell, s);
        }
    }

    /* No explicit sync: caller (CVODE RHS) will issue the next CUDA op
     * on stream 0, which provides ordering. */
}
