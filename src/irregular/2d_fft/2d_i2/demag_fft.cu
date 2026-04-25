/*
 * demag_fft.cu  —  v3: D2Z/Z2D real-to-complex FFT (half-spectrum).
 *
 * ─── Pipeline per Apply (every f() call) ─────────────────────────────
 *  1. gather_real_kernel: y_dev (SoA real) → d_m_real (3 × ny×nx doubles)
 *  2. cufftExecD2Z (batch=3, d_m_real → d_m_hat) → M̂ (half-spectrum)
 *  3. multiply_kernel:  Ĥ = f̂ · M̂ (read d_m_hat, write d_h_hat, ncell_half)
 *  4. cufftExecZ2D (batch=3, d_h_hat → d_h_real) → H (real)
 *  5. scatter_real_kernel: d_h_real + FFT-shift + scale → h_out_dev (SoA)
 *
 * f̂ (tensor spectra, 9 arrays × ncell_half) is computed ONCE in Demag_Init
 * using D2Z and stored permanently on device. No host transfers during Apply.
 *
 * Memory savings vs Z2Z: tensor uses ~50% less device memory.
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
__global__ static void gather_real_kernel(
    const double* __restrict__ y_dev,
    double*       __restrict__ d_m_real,
    int ncell)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncell) return;
    d_m_real[idx]           = y_dev[idx];
    d_m_real[ncell + idx]   = y_dev[ncell + idx];
    d_m_real[2*ncell + idx] = y_dev[2*ncell + idx];
}

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
    const cufftDoubleComplex* __restrict__ d_m_hat,
    cufftDoubleComplex*       __restrict__ d_h_hat,
    int ncell_half)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= ncell_half) return;

    const cufftDoubleComplex ma = d_m_hat[k];
    const cufftDoubleComplex mb = d_m_hat[ncell_half + k];
    const cufftDoubleComplex mc = d_m_hat[2*ncell_half + k];

#define CMUL_R(F,M) ((F)[k].x*(M).x - (F)[k].y*(M).y)
#define CMUL_I(F,M) ((F)[k].x*(M).y + (F)[k].y*(M).x)

    d_h_hat[k].x                = CMUL_R(dofaa,ma) + CMUL_R(dofab,mb) + CMUL_R(dofac,mc);
    d_h_hat[k].y                = CMUL_I(dofaa,ma) + CMUL_I(dofab,mb) + CMUL_I(dofac,mc);

    d_h_hat[ncell_half+k].x     = CMUL_R(dofba,ma) + CMUL_R(dofbb,mb) + CMUL_R(dofbc,mc);
    d_h_hat[ncell_half+k].y     = CMUL_I(dofba,ma) + CMUL_I(dofbb,mb) + CMUL_I(dofbc,mc);

    d_h_hat[2*ncell_half+k].x   = CMUL_R(dofca,ma) + CMUL_R(dofcb,mb) + CMUL_R(dofcc,mc);
    d_h_hat[2*ncell_half+k].y   = CMUL_I(dofca,ma) + CMUL_I(dofcb,mb) + CMUL_I(dofcc,mc);

#undef CMUL_R
#undef CMUL_I
}

__global__ static void scatter_real_kernel(
    const double* __restrict__ d_h_real,
    double*       __restrict__ h_out,
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

    h_out[dst]           = d_h_real[src]           * scale;
    h_out[ncell + dst]   = d_h_real[ncell + src]   * scale;
    h_out[2*ncell + dst] = d_h_real[2*ncell + src] * scale;
}

/* ── DemagData ─────────────────────────────────────────────────────── */
struct DemagData {
    int    nx, ny, ncell;
    int    ncell_half;       /* ny × (nx/2 + 1) — half-spectrum size */
    int    nx2, ny2;
    double strength;

    double Nxx0_scaled, Nyy0_scaled, Nzz0_scaled;

    cufftHandle plan_d2z_single;
    cufftHandle plan_d2z_batch;
    cufftHandle plan_z2d_batch;

    cufftDoubleComplex *dofaa, *dofab, *dofac;
    cufftDoubleComplex *dofba, *dofbb, *dofbc;
    cufftDoubleComplex *dofca, *dofcb, *dofcc;

    double             *d_m_real, *d_h_real;
    cufftDoubleComplex *d_m_hat,  *d_h_hat;
};

DemagData* Demag_Init(int nx, int ny, double thick, double demag_strength)
{
    printf("[Demag GPU v3] Init: nx=%d ny=%d thick=%.4f strength=%.4f\n",
           nx, ny, thick, demag_strength);
    printf("[Demag GPU v3] Real-to-complex FFT (D2Z/Z2D): half-spectrum layout.\n");
    printf("[Demag GPU v3] Batched (batch=3): forward and inverse in 2 calls/Apply.\n");

    DemagData *d = (DemagData*)calloc(1, sizeof(DemagData));
    if (!d) { fprintf(stderr,"[demag] calloc failed\n"); return NULL; }

    d->nx         = nx;
    d->ny         = ny;
    d->ncell      = nx * ny;
    d->ncell_half = ny * (nx / 2 + 1);
    d->nx2        = nx / 2;
    d->ny2        = ny / 2;
    d->strength   = demag_strength;

    const size_t rsz  = (size_t)(nx * ny) * sizeof(double);
    const size_t rsz3 = (size_t)3 * rsz;
    const size_t csz_half  = (size_t)d->ncell_half * sizeof(cufftDoubleComplex);
    const size_t csz_half3 = (size_t)3 * csz_half;

    printf("[Demag GPU v3] ncell=%d, ncell_half=%d (%.1f%% of full spectrum)\n",
           d->ncell, d->ncell_half, 100.0 * d->ncell_half / d->ncell);

    printf("[Demag GPU v3] Computing Newell tensor (calt/ctt, 81-pt avg)...\n");

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

    {
        const int i0 = d->ny2 * nx + d->nx2;
        d->Nxx0_scaled = taa[i0] * demag_strength;
        d->Nyy0_scaled = tbb[i0] * demag_strength;
        d->Nzz0_scaled = tcc[i0] * demag_strength;
        printf("[Demag GPU v3] N(0) = diag(%.4e, %.4e, %.4e)  [unscaled]\n",
               taa[i0], tbb[i0], tcc[i0]);
        printf("[Demag GPU v3] N(0) off-diag at origin: tab=%.2e tac=%.2e tbc=%.2e\n",
               tab[i0], tac[i0], tbc[i0]);
    }

    if (cufftPlan2d(&d->plan_d2z_single, ny, nx, CUFFT_D2Z) != CUFFT_SUCCESS) {
        fprintf(stderr,"[demag] cufftPlan2d (D2Z single) failed\n");
        free(taa);free(tab);free(tac);free(tba);free(tbb);
        free(tbc);free(tca);free(tcb);free(tcc);free(d);
        return NULL;
    }

    {
        int n[2] = {ny, nx};
        int inembed[2] = {ny, nx};
        int onembed[2] = {ny, nx/2 + 1};
        if (cufftPlanMany(&d->plan_d2z_batch, 2, n,
                          inembed, 1, d->ncell,
                          onembed, 1, d->ncell_half,
                          CUFFT_D2Z, 3) != CUFFT_SUCCESS) {
            fprintf(stderr,"[demag] cufftPlanMany (D2Z batch) failed\n");
            cufftDestroy(d->plan_d2z_single);
            free(taa);free(tab);free(tac);free(tba);free(tbb);
            free(tbc);free(tca);free(tcb);free(tcc);free(d);
            return NULL;
        }
    }

    {
        int n[2] = {ny, nx};
        int inembed[2] = {ny, nx/2 + 1};
        int onembed[2] = {ny, nx};
        if (cufftPlanMany(&d->plan_z2d_batch, 2, n,
                          inembed, 1, d->ncell_half,
                          onembed, 1, d->ncell,
                          CUFFT_Z2D, 3) != CUFFT_SUCCESS) {
            fprintf(stderr,"[demag] cufftPlanMany (Z2D batch) failed\n");
            cufftDestroy(d->plan_d2z_single);
            cufftDestroy(d->plan_d2z_batch);
            free(taa);free(tab);free(tac);free(tba);free(tbb);
            free(tbc);free(tca);free(tcb);free(tcc);free(d);
            return NULL;
        }
    }

    cudaMalloc((void**)&d->dofaa, csz_half); cudaMalloc((void**)&d->dofab, csz_half);
    cudaMalloc((void**)&d->dofac, csz_half); cudaMalloc((void**)&d->dofba, csz_half);
    cudaMalloc((void**)&d->dofbb, csz_half); cudaMalloc((void**)&d->dofbc, csz_half);
    cudaMalloc((void**)&d->dofca, csz_half); cudaMalloc((void**)&d->dofcb, csz_half);
    cudaMalloc((void**)&d->dofcc, csz_half);

    cudaMalloc((void**)&d->d_m_real, rsz3);
    cudaMalloc((void**)&d->d_h_real, rsz3);
    cudaMalloc((void**)&d->d_m_hat,  csz_half3);
    cudaMalloc((void**)&d->d_h_hat,  csz_half3);

    {
        auto fft_one_tensor = [&](double *t_host, cufftDoubleComplex *d_out) {
            cudaMemcpy(d->d_m_real, t_host, rsz, cudaMemcpyHostToDevice);
            cufftExecD2Z(d->plan_d2z_single, d->d_m_real, d->d_m_hat);
            cudaMemcpy(d_out, d->d_m_hat, csz_half, cudaMemcpyDeviceToDevice);
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
    }

    free(taa);free(tab);free(tac);free(tba);free(tbb);
    free(tbc);free(tca);free(tcb);free(tcc);

    cufftDestroy(d->plan_d2z_single);
    d->plan_d2z_single = 0;

    printf("[Demag GPU v3] Device: %.1f MB (tensor spectra) + %.1f MB (scratch)\n",
           (double)9 * csz_half / 1e6, (double)(rsz3 + rsz3 + csz_half3 + csz_half3) / 1e6);
    printf("[Demag GPU v3] Memory saving vs Z2Z: tensor %.1f%%, total scratch similar.\n",
           100.0 * (1.0 - (double)d->ncell_half / d->ncell));
    printf("[Demag GPU v3] Ready.\n");
    return d;
}

void Demag_Apply(DemagData *d, const double *y_dev, double *h_out_dev)
{
    if (!d) return;

    const int nx         = d->nx;
    const int ny         = d->ny;
    const int ncell      = d->ncell;
    const int ncell_half = d->ncell_half;
    const double scale   = d->strength / (double)(nx * ny);

    const int BLK = 256;

    {
        const int GRD = (ncell + BLK - 1) / BLK;
        gather_real_kernel<<<GRD, BLK>>>(y_dev, d->d_m_real, ncell);
    }

    cufftExecD2Z(d->plan_d2z_batch, d->d_m_real, d->d_m_hat);

    {
        const int GRD = (ncell_half + BLK - 1) / BLK;
        multiply_kernel<<<GRD, BLK>>>(
            d->dofaa, d->dofab, d->dofac,
            d->dofba, d->dofbb, d->dofbc,
            d->dofca, d->dofcb, d->dofcc,
            d->d_m_hat, d->d_h_hat, ncell_half);
    }

    cufftExecZ2D(d->plan_z2d_batch, d->d_h_hat, d->d_h_real);

    {
        dim3 block(16, 16);
        dim3 grid((nx + block.x - 1) / block.x,
                  (ny + block.y - 1) / block.y);
        scatter_real_kernel<<<grid, block>>>(
            d->d_h_real, h_out_dev, scale,
            nx, ny, ncell, d->nx2, d->ny2);
    }
}

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

void Demag_Destroy(DemagData *d)
{
    if (!d) return;

    if (d->plan_d2z_batch) cufftDestroy(d->plan_d2z_batch);
    if (d->plan_z2d_batch) cufftDestroy(d->plan_z2d_batch);

    cudaFree(d->dofaa); cudaFree(d->dofab); cudaFree(d->dofac);
    cudaFree(d->dofba); cudaFree(d->dofbb); cudaFree(d->dofbc);
    cudaFree(d->dofca); cudaFree(d->dofcb); cudaFree(d->dofcc);

    cudaFree(d->d_m_real); cudaFree(d->d_h_real);
    cudaFree(d->d_m_hat);  cudaFree(d->d_h_hat);

    free(d);
}
