/*
 * demag_fft.cu  —  GPU-only FFT demag pipeline.
 *
 * Replaces the old v2 that still did 4×D2H + 4×H2D per Apply (despite its
 * comments).  Now truly GPU-resident:
 *
 *   Demag_Init (once, at startup):
 *     1. calt/ctt          -> taa..tcc  (CPU, one-time cost)
 *     2. pack real -> complex (imag=0) (CPU)
 *     3. H2D tensor to temporary device buffers
 *     4. cufftExecZ2Z FORWARD on each -> dofaa..dofcc (PERMANENT on device)
 *     5. free tmp tensor, keep only the 9 spectra
 *     6. read Nxx(0), Nyy(0), Nzz(0) for preconditioner use
 *
 *   Demag_Apply (every f() call):
 *     7. pack_m_kernel:   y_dev (SoA) -> dima, dimb, dimc  (on device)
 *     8. cufftExecZ2Z FORWARD on each -> doma, domb, domc
 *     9. multiply_kernel: ĥ_α = Σ_β f̂_αβ · m̂_β
 *    10. cufftExecZ2Z INVERSE          -> drha, drhb, drhc (complex)
 *    11. unshift_h_kernel: real part + FFT-shift + (strength/N) scale
 *                          -> h_out_dev (SoA, 3*ncell, OVERWRITTEN)
 *
 *   Index shift (step 11):
 *     Tensor is stored centered at (nx/2, ny/2).  For properly-convolved
 *     output we need the standard circular-shift:
 *         h[i,j] = IFFT_result[((i+nx2) mod nx), ((j+ny2) mod ny)]
 *     (NOT the reflection formula from the old code.)
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

/* pack_m_kernel: SoA real y -> 3 complex arrays (imag=0).
 * y_dev layout:  [mx 0..ncell) | [my 0..ncell) | [mz 0..ncell)            */
__global__ static void pack_m_kernel(
    const double*            __restrict__ y_dev,
    cufftDoubleComplex*      __restrict__ dima,
    cufftDoubleComplex*      __restrict__ dimb,
    cufftDoubleComplex*      __restrict__ dimc,
    int ncell)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncell) return;
    dima[idx].x = y_dev[idx];              dima[idx].y = 0.0;
    dimb[idx].x = y_dev[ncell + idx];      dimb[idx].y = 0.0;
    dimc[idx].x = y_dev[2*ncell + idx];    dimc[idx].y = 0.0;
}

/* multiply_kernel: ĥ_α(k) = Σ_β f̂_αβ(k) · m̂_β(k)                         */
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
    const cufftDoubleComplex* __restrict__ doma,
    const cufftDoubleComplex* __restrict__ domb,
    const cufftDoubleComplex* __restrict__ domc,
    cufftDoubleComplex*       __restrict__ dkha,
    cufftDoubleComplex*       __restrict__ dkhb,
    cufftDoubleComplex*       __restrict__ dkhc,
    int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

#define CMUL_R(A,B) ((A)[k].x*(B)[k].x - (A)[k].y*(B)[k].y)
#define CMUL_I(A,B) ((A)[k].x*(B)[k].y + (A)[k].y*(B)[k].x)

    dkha[k].x = CMUL_R(dofaa,doma) + CMUL_R(dofab,domb) + CMUL_R(dofac,domc);
    dkha[k].y = CMUL_I(dofaa,doma) + CMUL_I(dofab,domb) + CMUL_I(dofac,domc);

    dkhb[k].x = CMUL_R(dofba,doma) + CMUL_R(dofbb,domb) + CMUL_R(dofbc,domc);
    dkhb[k].y = CMUL_I(dofba,doma) + CMUL_I(dofbb,domb) + CMUL_I(dofbc,domc);

    dkhc[k].x = CMUL_R(dofca,doma) + CMUL_R(dofcb,domb) + CMUL_R(dofcc,domc);
    dkhc[k].y = CMUL_I(dofca,doma) + CMUL_I(dofcb,domb) + CMUL_I(dofcc,domc);

#undef CMUL_R
#undef CMUL_I
}

/* unshift_h_kernel:
 *   src = (sj*nx + si) with  si = (i + nx2) mod nx,  sj = (j + ny2) mod ny
 *   h_out[j*nx+i]              = Re(drha[src]) * scale
 *   h_out[ncell + j*nx+i]      = Re(drhb[src]) * scale
 *   h_out[2*ncell + j*nx+i]    = Re(drhc[src]) * scale
 *
 * scale = strength / (nx*ny)  (the 1/N is cuFFT's inverse normalization).
 *
 * This OVERWRITES h_out (no accumulation) — simpler and correct.  */
__global__ static void unshift_h_kernel(
    const cufftDoubleComplex* __restrict__ drha,
    const cufftDoubleComplex* __restrict__ drhb,
    const cufftDoubleComplex* __restrict__ drhc,
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

    h_out[dst]           = drha[src].x * scale;
    h_out[ncell + dst]   = drhb[src].x * scale;
    h_out[2*ncell + dst] = drhc[src].x * scale;
}

/* ── DemagData ─────────────────────────────────────────────────────── */
struct DemagData {
    int    nx, ny, ncell;
    int    nx2, ny2;
    double strength;

    /* self-coupling (for preconditioner), already scaled by strength */
    double Nxx0_scaled, Nyy0_scaled, Nzz0_scaled;

    cufftHandle plan;

    /* permanent on device: tensor spectra */
    cufftDoubleComplex *dofaa, *dofab, *dofac;
    cufftDoubleComplex *dofba, *dofbb, *dofbc;
    cufftDoubleComplex *dofca, *dofcb, *dofcc;

    /* scratch on device: M FFT input/output */
    cufftDoubleComplex *dima, *doma;
    cufftDoubleComplex *dimb, *domb;
    cufftDoubleComplex *dimc, *domc;

    /* scratch on device: H̃ and IFFT output */
    cufftDoubleComplex *dkha, *drha;
    cufftDoubleComplex *dkhb, *drhb;
    cufftDoubleComplex *dkhc, *drhc;
};

/* ── Demag_Init (run once) ─────────────────────────────────────────── */
DemagData* Demag_Init(int nx, int ny, double thick, double demag_strength)
{
    printf("[Demag GPU] Init: nx=%d ny=%d thick=%.4f strength=%.4f\n",
           nx, ny, thick, demag_strength);
    printf("[Demag GPU] Fully on-device pipeline: pack -> FFT -> mul -> IFFT -> unshift.\n");
    printf("[Demag GPU] No per-step host transfers.\n");

    DemagData *d = (DemagData*)calloc(1, sizeof(DemagData));
    if (!d) { fprintf(stderr,"[demag] calloc failed\n"); return NULL; }

    d->nx       = nx;
    d->ny       = ny;
    d->ncell    = nx * ny;
    d->nx2      = nx / 2;
    d->ny2      = ny / 2;
    d->strength = demag_strength;

    const size_t csz = (size_t)(nx * ny) * sizeof(cufftDoubleComplex);

    /* calt on CPU */
    printf("[Demag GPU] Computing Newell tensor (calt/ctt, 81-pt avg)...\n");

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
        printf("[Demag GPU] N(0) = diag(%.4e, %.4e, %.4e)  [unscaled]\n",
               taa[i0], tbb[i0], tcc[i0]);
        printf("[Demag GPU] N(0) off-diag at origin: tab=%.2e tac=%.2e tbc=%.2e\n",
               tab[i0], tac[i0], tbc[i0]);
    }

    /* pack real tensor -> complex (imag=0) on host for H2D */
    cufftDoubleComplex *hfaa=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfab=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfac=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfba=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfbb=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfbc=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfca=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfcb=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfcc=(cufftDoubleComplex*)malloc(csz);

    for (int idx=0; idx < nx*ny; idx++) {
        hfaa[idx].x=taa[idx]; hfaa[idx].y=0.0;
        hfab[idx].x=tab[idx]; hfab[idx].y=0.0;
        hfac[idx].x=tac[idx]; hfac[idx].y=0.0;
        hfba[idx].x=tba[idx]; hfba[idx].y=0.0;
        hfbb[idx].x=tbb[idx]; hfbb[idx].y=0.0;
        hfbc[idx].x=tbc[idx]; hfbc[idx].y=0.0;
        hfca[idx].x=tca[idx]; hfca[idx].y=0.0;
        hfcb[idx].x=tcb[idx]; hfcb[idx].y=0.0;
        hfcc[idx].x=tcc[idx]; hfcc[idx].y=0.0;
    }
    free(taa);free(tab);free(tac);free(tba);free(tbb);
    free(tbc);free(tca);free(tcb);free(tcc);

    /* cuFFT plan */
    if (cufftPlan2d(&d->plan, ny, nx, CUFFT_Z2Z) != CUFFT_SUCCESS) {
        fprintf(stderr,"[demag] cufftPlan2d failed\n");
        free(hfaa);free(hfab);free(hfac);free(hfba);free(hfbb);
        free(hfbc);free(hfca);free(hfcb);free(hfcc);free(d);
        return NULL;
    }

    /* allocate permanent f̂ device arrays */
    cudaMalloc((void**)&d->dofaa,csz); cudaMalloc((void**)&d->dofab,csz);
    cudaMalloc((void**)&d->dofac,csz); cudaMalloc((void**)&d->dofba,csz);
    cudaMalloc((void**)&d->dofbb,csz); cudaMalloc((void**)&d->dofbc,csz);
    cudaMalloc((void**)&d->dofca,csz); cudaMalloc((void**)&d->dofcb,csz);
    cudaMalloc((void**)&d->dofcc,csz);

    /* temporary input buffers */
    cufftDoubleComplex *difaa,*difab,*difac,*difba,*difbb,*difbc,*difca,*difcb,*difcc;
    cudaMalloc((void**)&difaa,csz); cudaMalloc((void**)&difab,csz);
    cudaMalloc((void**)&difac,csz); cudaMalloc((void**)&difba,csz);
    cudaMalloc((void**)&difbb,csz); cudaMalloc((void**)&difbc,csz);
    cudaMalloc((void**)&difca,csz); cudaMalloc((void**)&difcb,csz);
    cudaMalloc((void**)&difcc,csz);

    cudaMemcpy(difaa,hfaa,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difab,hfab,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difac,hfac,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difba,hfba,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difbb,hfbb,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difbc,hfbc,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difca,hfca,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difcb,hfcb,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difcc,hfcc,csz,cudaMemcpyHostToDevice);

    free(hfaa);free(hfab);free(hfac);free(hfba);free(hfbb);
    free(hfbc);free(hfca);free(hfcb);free(hfcc);

    /* FFT forward -> dofXX permanent on device */
    cufftExecZ2Z(d->plan,difaa,d->dofaa,CUFFT_FORWARD);
    cufftExecZ2Z(d->plan,difab,d->dofab,CUFFT_FORWARD);
    cufftExecZ2Z(d->plan,difac,d->dofac,CUFFT_FORWARD);
    cufftExecZ2Z(d->plan,difba,d->dofba,CUFFT_FORWARD);
    cufftExecZ2Z(d->plan,difbb,d->dofbb,CUFFT_FORWARD);
    cufftExecZ2Z(d->plan,difbc,d->dofbc,CUFFT_FORWARD);
    cufftExecZ2Z(d->plan,difca,d->dofca,CUFFT_FORWARD);
    cufftExecZ2Z(d->plan,difcb,d->dofcb,CUFFT_FORWARD);
    cufftExecZ2Z(d->plan,difcc,d->dofcc,CUFFT_FORWARD);
    cudaDeviceSynchronize();

    cudaFree(difaa);cudaFree(difab);cudaFree(difac);
    cudaFree(difba);cudaFree(difbb);cudaFree(difbc);
    cudaFree(difca);cudaFree(difcb);cudaFree(difcc);

    /* persistent per-timestep scratch on device */
    cudaMalloc((void**)&d->dima,csz); cudaMalloc((void**)&d->doma,csz);
    cudaMalloc((void**)&d->dimb,csz); cudaMalloc((void**)&d->domb,csz);
    cudaMalloc((void**)&d->dimc,csz); cudaMalloc((void**)&d->domc,csz);
    cudaMalloc((void**)&d->dkha,csz); cudaMalloc((void**)&d->drha,csz);
    cudaMalloc((void**)&d->dkhb,csz); cudaMalloc((void**)&d->drhb,csz);
    cudaMalloc((void**)&d->dkhc,csz); cudaMalloc((void**)&d->drhc,csz);

    printf("[Demag GPU] Device: %.1f MB (tensor spectra) + %.1f MB (scratch)\n",
           (double)9 * csz / 1e6, (double)12 * csz / 1e6);
    printf("[Demag GPU] Ready.\n");
    return d;
}

/* ── Demag_Apply (per RHS call, fully GPU) ─────────────────────────── */
void Demag_Apply(DemagData *d, const double *y_dev, double *h_out_dev)
{
    if (!d) return;

    const int nx    = d->nx;
    const int ny    = d->ny;
    const int ncell = d->ncell;
    const double scale = d->strength / (double)(nx * ny);

    /* step 7: pack y (SoA real) -> complex on device */
    {
        const int b = 256;
        const int g = (ncell + b - 1) / b;
        pack_m_kernel<<<g, b>>>(y_dev, d->dima, d->dimb, d->dimc, ncell);
    }

    /* step 8: FFT forward (device-to-device) */
    cufftExecZ2Z(d->plan, d->dima, d->doma, CUFFT_FORWARD);
    cufftExecZ2Z(d->plan, d->dimb, d->domb, CUFFT_FORWARD);
    cufftExecZ2Z(d->plan, d->dimc, d->domc, CUFFT_FORWARD);

    /* step 9: ĥ = f̂ · m̂ (all on device) */
    {
        const int b = 256;
        const int g = (ncell + b - 1) / b;
        multiply_kernel<<<g, b>>>(
            d->dofaa, d->dofab, d->dofac,
            d->dofba, d->dofbb, d->dofbc,
            d->dofca, d->dofcb, d->dofcc,
            d->doma,  d->domb,  d->domc,
            d->dkha,  d->dkhb,  d->dkhc,
            ncell);
    }

    /* step 10: FFT inverse */
    cufftExecZ2Z(d->plan, d->dkha, d->drha, CUFFT_INVERSE);
    cufftExecZ2Z(d->plan, d->dkhb, d->drhb, CUFFT_INVERSE);
    cufftExecZ2Z(d->plan, d->dkhc, d->drhc, CUFFT_INVERSE);

    /* step 11: real-part + FFT-shift + scale (1/N included) -> SoA h_out */
    {
        dim3 block(16, 16);
        dim3 grid((nx + block.x - 1) / block.x,
                  (ny + block.y - 1) / block.y);
        unshift_h_kernel<<<grid, block>>>(
            d->drha, d->drhb, d->drhc,
            h_out_dev,
            scale,
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
    cufftDestroy(d->plan);

    cudaFree(d->dofaa); cudaFree(d->dofab); cudaFree(d->dofac);
    cudaFree(d->dofba); cudaFree(d->dofbb); cudaFree(d->dofbc);
    cudaFree(d->dofca); cudaFree(d->dofcb); cudaFree(d->dofcc);

    cudaFree(d->dima); cudaFree(d->doma);
    cudaFree(d->dimb); cudaFree(d->domb);
    cudaFree(d->dimc); cudaFree(d->domc);
    cudaFree(d->dkha); cudaFree(d->drha);
    cudaFree(d->dkhb); cudaFree(d->drhb);
    cudaFree(d->dkhc); cudaFree(d->drhc);

    free(d);
}
