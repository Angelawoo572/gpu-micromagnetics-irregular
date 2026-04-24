/*
 * demag_fft.cu  —  v2: batched cuFFT pipeline.
 *
 * ─── What changed from v1 ────────────────────────────────────────────
 * Switched from 6 individual cufftExecZ2Z calls per Apply (3 forward +
 * 3 inverse) to 2 batched calls (cufftPlanMany with batch=3).
 *
 * Nsight profile showed cuFFT was 23.6% of GPU kernel time (3.42 s).
 * Each of the 6 calls launches 2 sub-kernels (regular_fft + vector_fft),
 * so 12 kernel launches per f() call × 27,631 calls = 331,572 launches.
 *
 * Batching reduces this to 4 launches per f() call (2 batched calls ×
 * 2 sub-kernels each) = 110,524 launches — a 3× reduction in FFT launch
 * overhead.  cuFFT can also internally optimize memory access patterns
 * for the 3-batch case.
 *
 * Memory layout change:
 *   Old: 12 separate cufftDoubleComplex arrays (dim/dom/dkh/drh × a/b/c)
 *   New: 2 contiguous arrays of 3*ncell complex elements each:
 *     d_m_io:  pack M here → FFT forward in-place → contains M̂
 *     d_h_io:  multiply writes Ĥ here → FFT inverse in-place → contains H
 *   Scratch memory: 2 × 3*ncell × 16 = 6.3 MB (was 12 × ncell × 16 = 12.6 MB)
 *
 * ─── Pipeline per Apply (every f() call) ─────────────────────────────
 *  1. pack_m_kernel:    y_dev (SoA real) → d_m_io (3 contiguous complex)
 *  2. cufftExecZ2Z FORWARD (batch=3, in-place on d_m_io) → M̂
 *  3. multiply_kernel:  Ĥ = f̂ · M̂ (read d_m_io, write d_h_io)
 *  4. cufftExecZ2Z INVERSE (batch=3, in-place on d_h_io) → H
 *  5. unshift_h_kernel: real-part + FFT-shift + scale → h_out_dev (SoA)
 *
 * f̂ (tensor spectra, 9 arrays) is computed ONCE in Demag_Init and stored
 * permanently on device.  No host transfers during Apply.
 */

#include "demag_fft.h"

#include <cufft.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ── ctt: closed-form Newell integral (1-indexed dm[]) ──────────────── */
static void ctt(double b, double a, double sx, double sy, double dm[])
{
    double sz = 0.0;
    double xn=sx-a, xp=sx+a, yn=sy-a, yp=sy+a, zn=sz-b, zp=sz+b;
    double xn2=xn*xn, xp2=xp*xp, yn2=yn*yn, yp2=yp*yp, zn2=zn*zn, zp2=zp*zp;
    double dnnn=std::sqrt(xn2+yn2+zn2), dpnn=std::sqrt(xp2+yn2+zn2);
    double dnpn=std::sqrt(xn2+yp2+zn2), dnnp=std::sqrt(xn2+yn2+zp2);
    double dppn=std::sqrt(xp2+yp2+zn2), dnpp=std::sqrt(xn2+yp2+zp2);
    double dpnp=std::sqrt(xp2+yn2+zp2), dppp=std::sqrt(xp2+yp2+zp2);

    dm[1] = std::atan(zn*yn/(xn*dnnn))-std::atan(zp*yn/(xn*dnnp))
           -std::atan(zn*yp/(xn*dnpn))+std::atan(zp*yp/(xn*dnpp))
           -std::atan(zn*yn/(xp*dpnn))+std::atan(zp*yn/(xp*dpnp))
           +std::atan(zn*yp/(xp*dppn))-std::atan(zp*yp/(xp*dppp));
    dm[2] = std::log((dnnn-zn)/(dnnp-zp))-std::log((dpnn-zn)/(dpnp-zp))
           -std::log((dnpn-zn)/(dnpp-zp))+std::log((dppn-zn)/(dppp-zp));
    dm[3] = std::log((dnnn-yn)/(dnpn-yp))-std::log((dpnn-yn)/(dppn-yp))
           -std::log((dnnp-yn)/(dnpp-yp))+std::log((dpnp-yn)/(dppp-yp));
    dm[4] = std::log((dnnn-zn)/(dnnp-zp))-std::log((dnpn-zn)/(dnpp-zp))
           -std::log((dpnn-zn)/(dpnp-zp))+std::log((dppn-zn)/(dppp-zp));
    dm[5] = std::atan(zn*xn/(yn*dnnn))-std::atan(zp*xn/(yn*dnnp))
           -std::atan(zn*xp/(yn*dpnn))+std::atan(zp*xp/(yn*dpnp))
           -std::atan(zn*xn/(yp*dnpn))+std::atan(zp*xn/(yp*dnpp))
           +std::atan(zn*xp/(yp*dppn))-std::atan(zp*xp/(yp*dppp));
    dm[6] = std::log((dnnn-xn)/(dpnn-xp))-std::log((dnpn-xn)/(dppn-xp))
           -std::log((dnnp-xn)/(dpnp-xp))+std::log((dnpp-xn)/(dppp-xp));
    dm[7] = std::log((dnnn-yn)/(dnpn-yp))-std::log((dnnp-yn)/(dnpp-yp))
           -std::log((dpnn-yn)/(dppn-yp))+std::log((dpnp-yn)/(dppp-yp));
    dm[8] = std::log((dnnn-xn)/(dpnn-xp))-std::log((dnnp-xn)/(dpnp-xp))
           -std::log((dnpn-xn)/(dppn-xp))+std::log((dnpp-xn)/(dppp-xp));
    dm[9] = std::atan(xn*yn/(zn*dnnn))-std::atan(xp*yn/(zn*dpnn))
           -std::atan(xn*yp/(zn*dnpn))+std::atan(xp*yp/(zn*dppn))
           -std::atan(xn*yn/(zp*dnnp))+std::atan(xp*yn/(zp*dpnp))
           +std::atan(xn*yp/(zp*dnpp))-std::atan(xp*yp/(zp*dppp));
}

/* ── calt: 81-point averaging of ctt over cell face ─────────────────── */
static void calt(double thik, int mdx, int mdy,
                 double taa[], double tab[], double tac[],
                 double tba[], double tbb[], double tbc[],
                 double tca[], double tcb[], double tcc[])
{
    int mdx2=mdx/2, mdy2=mdy/2;
    double a=0.49999, b=0.5*thik;
    double dm[10];
    for (int j=0;j<mdy;j++) for (int i=0;i<mdx;i++) {
        int ikn=j*mdx+i;
        taa[ikn]=tab[ikn]=tac[ikn]=0.0;
        tba[ikn]=tbb[ikn]=tbc[ikn]=0.0;
        tca[ikn]=tcb[ikn]=tcc[ikn]=0.0;
        for (int jy=-4;jy<=4;jy++) {
            double sy=double(j-mdy2)+0.1*double(jy);
            for (int ix=-4;ix<=4;ix++) {
                double sx=double(i-mdx2)+0.1*double(ix);
                ctt(b,a,sx,sy,dm);
                taa[ikn]+=dm[1]; tab[ikn]+=dm[2]; tac[ikn]+=dm[3];
                tba[ikn]+=dm[4]; tbb[ikn]+=dm[5]; tbc[ikn]+=dm[6];
                tca[ikn]+=dm[7]; tcb[ikn]+=dm[8]; tcc[ikn]+=dm[9];
            }
        }
        taa[ikn]/=81.; tab[ikn]/=81.; tac[ikn]/=81.;
        tba[ikn]/=81.; tbb[ikn]/=81.; tbc[ikn]/=81.;
        tca[ikn]/=81.; tcb[ikn]/=81.; tcc[ikn]/=81.;
    }
}

/* ── Device kernels ────────────────────────────────────────────────── */

/* pack_m_kernel: SoA real y → 3 contiguous complex arrays (imag=0).
 * Layout in d_m_io:
 *   [0       .. ncell-1]   = mx + 0i
 *   [ncell   .. 2ncell-1]  = my + 0i
 *   [2*ncell .. 3ncell-1]  = mz + 0i
 */
__global__ static void pack_m_kernel(
    const double*            __restrict__ y_dev,
    cufftDoubleComplex*      __restrict__ d_m_io,
    int ncell)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncell) return;
    d_m_io[idx].x              = y_dev[idx];           d_m_io[idx].y              = 0.0;
    d_m_io[ncell + idx].x      = y_dev[ncell + idx];   d_m_io[ncell + idx].y      = 0.0;
    d_m_io[2*ncell + idx].x    = y_dev[2*ncell + idx]; d_m_io[2*ncell + idx].y    = 0.0;
}

/* multiply_kernel: Ĥ_α(k) = Σ_β f̂_αβ(k) · M̂_β(k)
 *
 * Reads M̂ from d_m_io (contiguous: [mx_hat | my_hat | mz_hat])
 * Writes Ĥ to d_h_io (contiguous: [hx_hat | hy_hat | hz_hat])
 * Tensor spectra f̂_αβ are separate (permanent on device).
 */
__global__ static void multiply_kernel(
    const cufftDoubleComplex* __restrict__ dofaa,
    const cufftDoubleComplex* __restrict__ dofab,
    const cufftDoubleComplex* __restrict__ dofac,
    const cufftDoubleComplex* __restrict__ dofba,
    const cufftDoubleComplex* __restrict__ dofbb,
    const cufftDoubleComplex* __restrict__ dofbc,
    const cufftDoubleComplex* __restrict__ dofca,
    const cufftDoubleComplex* __restrict__ dofcb,
    const cufftDoubleComplex* __restrict__ dofcc,
    const cufftDoubleComplex* __restrict__ d_m_io,     /* M̂: 3*ncell */
    cufftDoubleComplex*       __restrict__ d_h_io,     /* Ĥ: 3*ncell */
    int ncell)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= ncell) return;

    /* Read M̂ components from contiguous batch layout */
    const cufftDoubleComplex ma = d_m_io[k];
    const cufftDoubleComplex mb = d_m_io[ncell + k];
    const cufftDoubleComplex mc = d_m_io[2*ncell + k];

#define CMUL_R(F,M) ((F)[k].x*(M).x - (F)[k].y*(M).y)
#define CMUL_I(F,M) ((F)[k].x*(M).y + (F)[k].y*(M).x)

    /* Ĥ_x = f̂_xx M̂_x + f̂_xy M̂_y + f̂_xz M̂_z */
    d_h_io[k].x           = CMUL_R(dofaa,ma) + CMUL_R(dofab,mb) + CMUL_R(dofac,mc);
    d_h_io[k].y           = CMUL_I(dofaa,ma) + CMUL_I(dofab,mb) + CMUL_I(dofac,mc);

    /* Ĥ_y = f̂_yx M̂_x + f̂_yy M̂_y + f̂_yz M̂_z */
    d_h_io[ncell+k].x     = CMUL_R(dofba,ma) + CMUL_R(dofbb,mb) + CMUL_R(dofbc,mc);
    d_h_io[ncell+k].y     = CMUL_I(dofba,ma) + CMUL_I(dofbb,mb) + CMUL_I(dofbc,mc);

    /* Ĥ_z = f̂_zx M̂_x + f̂_zy M̂_y + f̂_zz M̂_z */
    d_h_io[2*ncell+k].x   = CMUL_R(dofca,ma) + CMUL_R(dofcb,mb) + CMUL_R(dofcc,mc);
    d_h_io[2*ncell+k].y   = CMUL_I(dofca,ma) + CMUL_I(dofcb,mb) + CMUL_I(dofcc,mc);

#undef CMUL_R
#undef CMUL_I
}

/* unshift_h_kernel:
 *   Reads d_h_io (contiguous: [hx | hy | hz], each ncell complex).
 *   FFT-shift: src_idx = ((i+nx2) mod nx, (j+ny2) mod ny)
 *   Writes to h_out_dev SoA (3*ncell real), scaled by strength/(nx*ny).
 */
__global__ static void unshift_h_kernel(
    const cufftDoubleComplex* __restrict__ d_h_io,
    double*                   __restrict__ h_out,
    double scale,
    int nx, int ny, int ncell, int nx2, int ny2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int dst = j*nx + i;
    int si  = (i + nx2) % nx;
    int sj  = (j + ny2) % ny;
    int src = sj*nx + si;

    h_out[dst]           = d_h_io[src].x           * scale;
    h_out[ncell + dst]   = d_h_io[ncell + src].x   * scale;
    h_out[2*ncell + dst] = d_h_io[2*ncell + src].x * scale;
}

/* ── DemagData ─────────────────────────────────────────────────────── */
struct DemagData {
    int    nx, ny, ncell;
    int    nx2, ny2;
    double strength;

    /* self-coupling (for preconditioner), already scaled by strength */
    double Nxx0_scaled, Nyy0_scaled, Nzz0_scaled;

    /* cuFFT plans */
    cufftHandle plan_single;      /* for init: 1 × (ny, nx) Z2Z */
    cufftHandle plan_batch;       /* for Apply: 3 × (ny, nx) Z2Z, stride ncell */

    /* permanent on device: tensor spectra (9 × ncell complex) */
    cufftDoubleComplex *dofaa, *dofab, *dofac;
    cufftDoubleComplex *dofba, *dofbb, *dofbc;
    cufftDoubleComplex *dofca, *dofcb, *dofcc;

    /* scratch on device: batched M and H (2 contiguous blocks) */
    cufftDoubleComplex *d_m_io;   /* 3*ncell: pack → FFT forward in-place → M̂ */
    cufftDoubleComplex *d_h_io;   /* 3*ncell: multiply → FFT inverse in-place → H */
};

/* ── Demag_Init (run once) ─────────────────────────────────────────── */
DemagData* Demag_Init(int nx, int ny, double thick, double demag_strength)
{
    printf("[Demag GPU v2] Init: nx=%d ny=%d thick=%.4f strength=%.4f\n",
           nx, ny, thick, demag_strength);
    printf("[Demag GPU v2] Batched cuFFT (batch=3): 2 calls/Apply instead of 6.\n");
    printf("[Demag GPU v2] In-place FFT, no per-step host transfers.\n");

    DemagData *d = (DemagData*)calloc(1, sizeof(DemagData));
    if (!d) { fprintf(stderr,"[demag] calloc failed\n"); return NULL; }

    d->nx       = nx;
    d->ny       = ny;
    d->ncell    = nx * ny;
    d->nx2      = nx / 2;
    d->ny2      = ny / 2;
    d->strength = demag_strength;

    const size_t csz  = (size_t)(nx * ny) * sizeof(cufftDoubleComplex);
    const size_t csz3 = (size_t)3 * csz;   /* batched buffer size */

    /* ── calt on CPU ──────────────────────────────────────────────── */
    printf("[Demag GPU v2] Computing Newell tensor (calt/ctt, 81-pt avg)...\n");

    double *taa=(double*)calloc(nx*ny,sizeof(double));
    double *tab=(double*)calloc(nx*ny,sizeof(double));
    double *tac=(double*)calloc(nx*ny,sizeof(double));
    double *tba=(double*)calloc(nx*ny,sizeof(double));
    double *tbb=(double*)calloc(nx*ny,sizeof(double));
    double *tbc=(double*)calloc(nx*ny,sizeof(double));
    double *tca=(double*)calloc(nx*ny,sizeof(double));
    double *tcb=(double*)calloc(nx*ny,sizeof(double));
    double *tcc=(double*)calloc(nx*ny,sizeof(double));

    if (!taa||!tab||!tac||!tba||!tbb||!tbc||!tca||!tcb||!tcc) {
        fprintf(stderr,"[demag] tensor alloc failed\n");
        free(taa);free(tab);free(tac);free(tba);free(tbb);
        free(tbc);free(tca);free(tcb);free(tcc);free(d);
        return NULL;
    }
    calt(thick, nx, ny, taa,tab,tac, tba,tbb,tbc, tca,tcb,tcc);

    /* extract N(0) — stored at index (ny2, nx2) in centered convention */
    {
        const int i0 = d->ny2 * nx + d->nx2;
        d->Nxx0_scaled = taa[i0] * demag_strength;
        d->Nyy0_scaled = tbb[i0] * demag_strength;
        d->Nzz0_scaled = tcc[i0] * demag_strength;
        printf("[Demag GPU v2] N(0) = diag(%.4e, %.4e, %.4e)  [unscaled]\n",
               taa[i0], tbb[i0], tcc[i0]);
        printf("[Demag GPU v2] N(0) off-diag at origin: tab=%.2e tac=%.2e tbc=%.2e\n",
               tab[i0], tac[i0], tbc[i0]);
    }

    /* ── cuFFT plans ─────────────────────────────────────────────── */
    /* Single plan: used during init to FFT the 9 tensor components. */
    if (cufftPlan2d(&d->plan_single, ny, nx, CUFFT_Z2Z) != CUFFT_SUCCESS) {
        fprintf(stderr,"[demag] cufftPlan2d failed\n");
        free(taa);free(tab);free(tac);free(tba);free(tbb);
        free(tbc);free(tca);free(tcb);free(tcc);free(d);
        return NULL;
    }

    /* Batched plan: used every Apply call — 3 transforms at once.
     * dims[0]=ny (rows), dims[1]=nx (cols).
     * stride=1: elements within one 2D array are contiguous.
     * dist=ncell: gap between batch 0 and batch 1 is ncell complex elements. */
    {
        int dims[2] = {ny, nx};
        if (cufftPlanMany(&d->plan_batch, 2, dims,
                          dims, 1, d->ncell,   /* inembed, istride, idist */
                          dims, 1, d->ncell,   /* onembed, ostride, odist */
                          CUFFT_Z2Z, 3) != CUFFT_SUCCESS) {
            fprintf(stderr,"[demag] cufftPlanMany (batch=3) failed\n");
            cufftDestroy(d->plan_single);
            free(taa);free(tab);free(tac);free(tba);free(tbb);
            free(tbc);free(tca);free(tcb);free(tcc);free(d);
            return NULL;
        }
    }

    /* ── Allocate permanent f̂ device arrays (9 × ncell) ──────────── */
    cudaMalloc((void**)&d->dofaa,csz); cudaMalloc((void**)&d->dofab,csz);
    cudaMalloc((void**)&d->dofac,csz); cudaMalloc((void**)&d->dofba,csz);
    cudaMalloc((void**)&d->dofbb,csz); cudaMalloc((void**)&d->dofbc,csz);
    cudaMalloc((void**)&d->dofca,csz); cudaMalloc((void**)&d->dofcb,csz);
    cudaMalloc((void**)&d->dofcc,csz);

    /* ── FFT tensor components one at a time (init only) ─────────── */
    /* Use d_m_io as temporary buffer (allocate it now). */
    cudaMalloc((void**)&d->d_m_io, csz3);
    cudaMalloc((void**)&d->d_h_io, csz3);

    /* Helper: pack one real tensor to complex on host, H2D, FFT, store. */
    {
        cufftDoubleComplex *h_tmp = (cufftDoubleComplex*)malloc(csz);
        /* We use d_m_io[0..ncell-1] as a temp device buffer for single FFTs. */
        cufftDoubleComplex *d_tmp_in  = d->d_m_io;            /* reuse batch buf */
        cufftDoubleComplex *d_tmp_out;                         /* permanent f̂ target */

        auto fft_one_tensor = [&](double *t_host, cufftDoubleComplex *d_out) {
            for (int idx = 0; idx < nx*ny; idx++) {
                h_tmp[idx].x = t_host[idx]; h_tmp[idx].y = 0.0;
            }
            cudaMemcpy(d_tmp_in, h_tmp, csz, cudaMemcpyHostToDevice);
            cufftExecZ2Z(d->plan_single, d_tmp_in, d_out, CUFFT_FORWARD);
        };

        fft_one_tensor(taa, d->dofaa);
        fft_one_tensor(tab, d->dofab);
        fft_one_tensor(tac, d->dofac);
        fft_one_tensor(tba, d->dofba);
        fft_one_tensor(tbb, d->dofbb);
        fft_one_tensor(tbc, d->dofbc);
        fft_one_tensor(tca, d->dofca);
        fft_one_tensor(tcb, d->dofcb);
        fft_one_tensor(tcc, d->dofcc);
        cudaDeviceSynchronize();

        free(h_tmp);
    }

    free(taa);free(tab);free(tac);free(tba);free(tbb);
    free(tbc);free(tca);free(tcb);free(tcc);

    /* plan_single is no longer needed after init — free it to save cuFFT state */
    cufftDestroy(d->plan_single);
    d->plan_single = 0;

    printf("[Demag GPU v2] Device: %.1f MB (tensor spectra) + %.1f MB (scratch)\n",
           (double)9 * csz / 1e6, (double)2 * csz3 / 1e6);
    printf("[Demag GPU v2] Ready.\n");
    return d;
}

/* ── Demag_Apply (per RHS call, fully GPU, batched FFT) ────────────── */
void Demag_Apply(DemagData *d, const double *y_dev, double *h_out_dev)
{
    if (!d) return;

    const int nx    = d->nx;
    const int ny    = d->ny;
    const int ncell = d->ncell;
    const double scale = d->strength / (double)(nx * ny);

    const int BLK = 256;
    const int GRD = (ncell + BLK - 1) / BLK;

    /* Step 1: pack SoA real y → 3 contiguous complex in d_m_io */
    pack_m_kernel<<<GRD, BLK>>>(y_dev, d->d_m_io, ncell);

    /* Step 2: FFT forward, batch=3, IN-PLACE on d_m_io
     * After this, d_m_io contains [M̂_x | M̂_y | M̂_z]. */
    cufftExecZ2Z(d->plan_batch, d->d_m_io, d->d_m_io, CUFFT_FORWARD);

    /* Step 3: Ĥ = f̂ · M̂ (read d_m_io, write d_h_io) */
    multiply_kernel<<<GRD, BLK>>>(
        d->dofaa, d->dofab, d->dofac,
        d->dofba, d->dofbb, d->dofbc,
        d->dofca, d->dofcb, d->dofcc,
        d->d_m_io, d->d_h_io, ncell);

    /* Step 4: FFT inverse, batch=3, IN-PLACE on d_h_io */
    cufftExecZ2Z(d->plan_batch, d->d_h_io, d->d_h_io, CUFFT_INVERSE);

    /* Step 5: real-part + FFT-shift + scale → SoA h_out_dev */
    {
        dim3 block(16, 16);
        dim3 grid((nx + block.x - 1) / block.x,
                  (ny + block.y - 1) / block.y);
        unshift_h_kernel<<<grid, block>>>(
            d->d_h_io, h_out_dev, scale,
            nx, ny, ncell, d->nx2, d->ny2);
    }
}

/* ── Demag_GetSelfCoupling ─────────────────────────────────────────── */
void Demag_GetSelfCoupling(DemagData *d,
                           double *nxx0, double *nyy0, double *nzz0)
{
    if (!d) {
        if (nxx0) *nxx0 = 0.0;
        if (nyy0) *nyy0 = 0.0;
        if (nzz0) *nzz0 = 0.0;
        return;
    }
    if (nxx0) *nxx0 = d->Nxx0_scaled;
    if (nyy0) *nyy0 = d->Nyy0_scaled;
    if (nzz0) *nzz0 = d->Nzz0_scaled;
}

/* ── Demag_Destroy ─────────────────────────────────────────────────── */
void Demag_Destroy(DemagData *d)
{
    if (!d) return;
    if (d->plan_batch) cufftDestroy(d->plan_batch);
    /* plan_single already destroyed in Init */

    cudaFree(d->dofaa); cudaFree(d->dofab); cudaFree(d->dofac);
    cudaFree(d->dofba); cudaFree(d->dofbb); cudaFree(d->dofbc);
    cudaFree(d->dofca); cudaFree(d->dofcb); cudaFree(d->dofcc);

    cudaFree(d->d_m_io);
    cudaFree(d->d_h_io);

    free(d);
}
