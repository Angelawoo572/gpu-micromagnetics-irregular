/*
 * demag_fft.cu
 * wrapping into Demag_Init/Apply/Destroy
 * for integration with CVODE f() — the data flow is identical.
 *
 *  flow:
 *   1. calt/ctt  → real-space tensor taa..tcc  (CPU)
 *   2. pack into complex host arrays hfaa..hfcc  (imag=0)  (CPU)
 *   3. H2D copy → difaa..difcc
 *   4. cufftExecZ2Z FORWARD → dofaa..dofcc (GPU)
 *   5. D2H copy → hofaa..hofcc (CPU, keep on host)
 *
 *   Per-timestep (Demag_Apply):
 *   6. pack M into complex hma_c..hmc_c (CPU, gather from SoA)
 *   7. H2D copy → dima..dimc
 *   8. cufftExecZ2Z FORWARD → doma..domc  (GPU)
 *   9. D2H copy → homa..homc (CPU)
 *  10. host multiply loop: hkha = cmul(hofaa,homa)+... (CPU — exactly as pseudocode)
 *  11. H2D copy hkha..hkhc → dkha..dkhc
 *  12. cufftExecZ2Z INVERSE → drha..drhc  (GPU)
 *  13. D2H copy → hrha..hrhc  (CPU)
 *  14. scatter hrha[idxnew].x/(nn*nn) back into SoA h_out  (CPU)
 *
 * Note: the index remapping in step 14 uses the  idxnew formula:
 *   if j < nn2:  jy = nn2 - j
 *   else:        jy = nn - j + nn2 - 1
 * (same for i/ix), then idxnew = jy*nn + ix
 */

#include "demag_fft.h"

#include <cufft.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static inline cufftDoubleComplex h_cadd(cufftDoubleComplex a, cufftDoubleComplex b)
{
    cufftDoubleComplex r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    return r;
}

static inline cufftDoubleComplex h_cmul(cufftDoubleComplex a, cufftDoubleComplex b)
{
    cufftDoubleComplex r;
    r.x = a.x * b.x - a.y * b.y;
    r.y = a.x * b.y + a.y * b.x;
    return r;
}

/* dm[] is 1-indexed: dm[1]..dm[9].  Size declared as dm[10] in calt.    */
static void ctt(double b, double a, double sx, double sy, double dm[])
{
    double sz = 0.0;

    double xn = sx - a;
    double xp = sx + a;
    double yn = sy - a;
    double yp = sy + a;
    double zn = sz - b;
    double zp = sz + b;

    double xn2 = xn * xn;
    double xp2 = xp * xp;
    double yn2 = yn * yn;
    double yp2 = yp * yp;
    double zn2 = zn * zn;
    double zp2 = zp * zp;

    double dnnn = std::sqrt(xn2 + yn2 + zn2);
    double dpnn = std::sqrt(xp2 + yn2 + zn2);
    double dnpn = std::sqrt(xn2 + yp2 + zn2);
    double dnnp = std::sqrt(xn2 + yn2 + zp2);
    double dppn = std::sqrt(xp2 + yp2 + zn2);
    double dnpp = std::sqrt(xn2 + yp2 + zp2);
    double dpnp = std::sqrt(xp2 + yn2 + zp2);
    double dppp = std::sqrt(xp2 + yp2 + zp2);

    dm[1] =
        std::atan(zn * yn / (xn * dnnn)) - std::atan(zp * yn / (xn * dnnp))
        - std::atan(zn * yp / (xn * dnpn)) + std::atan(zp * yp / (xn * dnpp))
        - std::atan(zn * yn / (xp * dpnn)) + std::atan(zp * yn / (xp * dpnp))
        + std::atan(zn * yp / (xp * dppn)) - std::atan(zp * yp / (xp * dppp));

    dm[2] =
        std::log((dnnn - zn) / (dnnp - zp)) - std::log((dpnn - zn) / (dpnp - zp))
        - std::log((dnpn - zn) / (dnpp - zp)) + std::log((dppn - zn) / (dppp - zp));

    dm[3] =
        std::log((dnnn - yn) / (dnpn - yp)) - std::log((dpnn - yn) / (dppn - yp))
        - std::log((dnnp - yn) / (dnpp - yp)) + std::log((dpnp - yn) / (dppp - yp));

    dm[4] =
        std::log((dnnn - zn) / (dnnp - zp)) - std::log((dnpn - zn) / (dnpp - zp))
        - std::log((dpnn - zn) / (dpnp - zp)) + std::log((dppn - zn) / (dppp - zp));

    dm[5] =
        std::atan(zn * xn / (yn * dnnn)) - std::atan(zp * xn / (yn * dnnp))
        - std::atan(zn * xp / (yn * dpnn)) + std::atan(zp * xp / (yn * dpnp))
        - std::atan(zn * xn / (yp * dnpn)) + std::atan(zp * xn / (yp * dnpp))
        + std::atan(zn * xp / (yp * dppn)) - std::atan(zp * xp / (yp * dppp));

    dm[6] =
        std::log((dnnn - xn) / (dpnn - xp)) - std::log((dnpn - xn) / (dppn - xp))
        - std::log((dnnp - xn) / (dpnp - xp)) + std::log((dnpp - xn) / (dppp - xp));

    dm[7] =
        std::log((dnnn - yn) / (dnpn - yp)) - std::log((dnnp - yn) / (dnpp - yp))
        - std::log((dpnn - yn) / (dppn - yp)) + std::log((dpnp - yn) / (dppp - yp));

    dm[8] =
        std::log((dnnn - xn) / (dpnn - xp)) - std::log((dnnp - xn) / (dpnp - xp))
        - std::log((dnpn - xn) / (dppn - xp)) + std::log((dnpp - xn) / (dppp - xp));

    dm[9] =
        std::atan(xn * yn / (zn * dnnn)) - std::atan(xp * yn / (zn * dpnn))
        - std::atan(xn * yp / (zn * dnpn)) + std::atan(xp * yp / (zn * dppn))
        - std::atan(xn * yn / (zp * dnnp)) + std::atan(xp * yn / (zp * dpnp))
        + std::atan(xn * yp / (zp * dnpp)) - std::atan(xp * yp / (zp * dppp));
}

static int calt(double thik, int mdx, int mdy,
                double taa[], double tab[], double tac[],
                double tba[], double tbb[], double tbc[],
                double tca[], double tcb[], double tcc[])
{
    int i, j, ix, jy, ikn;
    int mdx2 = mdx / 2;
    int mdy2 = mdy / 2;
    double sx, sy;
    double a = 0.49999;
    double b = 0.5 * thik;
    double dm[10];   /* 1-indexed: dm[1]..dm[9] */

    for (j = 0; j < mdy; j++) {
        for (i = 0; i < mdx; i++) {
            ikn = j * mdx + i;
            taa[ikn] = 0.0;
            tab[ikn] = 0.0;
            tac[ikn] = 0.0;
            tba[ikn] = 0.0;
            tbb[ikn] = 0.0;
            tbc[ikn] = 0.0;
            tca[ikn] = 0.0;
            tcb[ikn] = 0.0;
            tcc[ikn] = 0.0;

            for (jy = -4; jy <= 4; jy++) {
                sy = double(j - mdy2) + 0.1 * double(jy);
                for (ix = -4; ix <= 4; ix++) {
                    sx = double(i - mdx2) + 0.1 * double(ix);
                    ctt(b, a, sx, sy, dm);
                    taa[ikn] = taa[ikn] + dm[1];
                    tab[ikn] = tab[ikn] + dm[2];
                    tac[ikn] = tac[ikn] + dm[3];
                    tba[ikn] = tba[ikn] + dm[4];
                    tbb[ikn] = tbb[ikn] + dm[5];
                    tbc[ikn] = tbc[ikn] + dm[6];
                    tca[ikn] = tca[ikn] + dm[7];
                    tcb[ikn] = tcb[ikn] + dm[8];
                    tcc[ikn] = tcc[ikn] + dm[9];
                }
            }

            taa[ikn] = taa[ikn] / 81.0;
            tab[ikn] = tab[ikn] / 81.0;
            tac[ikn] = tac[ikn] / 81.0;
            tba[ikn] = tba[ikn] / 81.0;
            tbb[ikn] = tbb[ikn] / 81.0;
            tbc[ikn] = tbc[ikn] / 81.0;
            tca[ikn] = tca[ikn] / 81.0;
            tcb[ikn] = tcb[ikn] / 81.0;
            tcc[ikn] = tcc[ikn] / 81.0;
        }
    }
    return 1;
}

/* 
 * DemagData
 *  */
struct DemagData {
    int nx, ny, ncell;
    int nn2;
    double strength;

    cufftHandle plan;

    /* Host: FFT of tensor (kept on host for multiply loop) */
    cufftDoubleComplex *hofaa, *hofab, *hofac;
    cufftDoubleComplex *hofba, *hofbb, *hofbc;
    cufftDoubleComplex *hofca, *hofcb, *hofcc;

    /* Device: M FFT buffers */
    cufftDoubleComplex *dima, *dimb, *dimc;
    cufftDoubleComplex *doma, *domb, *domc;

    /* Host: M FFT output, multiply output, IFFT output */
    cufftDoubleComplex *homa, *homb, *homc;
    cufftDoubleComplex *hkha, *hkhb, *hkhc;
    cufftDoubleComplex *hrha, *hrhb, *hrhc;

    /* Device: H IFFT buffers */
    cufftDoubleComplex *dkha, *dkhb, *dkhc;
    cufftDoubleComplex *drha, *drhb, *drhc;
};

/* 
 * Demag_Init  (steps 1-5)
 *  */
DemagData* Demag_Init(int nx, int ny, double thick, double demag_strength)
{
    printf("[Demag] Init: nx=%d ny=%d thick=%.4f strength=%.4f\n",
           nx, ny, thick, demag_strength);

    DemagData *d = (DemagData*)calloc(1, sizeof(DemagData));
    if (!d) { fprintf(stderr,"[demag] calloc failed\n"); return NULL; }

    d->nx       = nx;
    d->ny       = ny;
    d->ncell    = nx * ny;
    d->nn2      = ny / 2;
    d->strength = demag_strength;

    const size_t csz = (size_t)(nx * ny) * sizeof(cufftDoubleComplex);

    /* step 1: calt */
    printf("[Demag] Computing calt (81-pt averaging)...\n");

    double *taa = (double*)calloc(nx*ny, sizeof(double));
    double *tab = (double*)calloc(nx*ny, sizeof(double));
    double *tac = (double*)calloc(nx*ny, sizeof(double));
    double *tba = (double*)calloc(nx*ny, sizeof(double));
    double *tbb = (double*)calloc(nx*ny, sizeof(double));
    double *tbc = (double*)calloc(nx*ny, sizeof(double));
    double *tca = (double*)calloc(nx*ny, sizeof(double));
    double *tcb = (double*)calloc(nx*ny, sizeof(double));
    double *tcc = (double*)calloc(nx*ny, sizeof(double));

    if (!taa||!tab||!tac||!tba||!tbb||!tbc||!tca||!tcb||!tcc) {
        fprintf(stderr,"[demag] tensor alloc failed\n");
        free(taa);free(tab);free(tac);free(tba);free(tbb);
        free(tbc);free(tca);free(tcb);free(tcc);free(d);
        return NULL;
    }
    calt(thick, nx, ny, taa,tab,tac, tba,tbb,tbc, tca,tcb,tcc);
    printf("[Demag] calt done.\n");

    /* step 2: pack into complex (imag=0) */
    cufftDoubleComplex *hfaa = (cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfab = (cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfac = (cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfba = (cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfbb = (cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfbc = (cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfca = (cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfcb = (cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfcc = (cufftDoubleComplex*)malloc(csz);

    for (int idx = 0; idx < nx*ny; idx++) {
        hfaa[idx].x = taa[idx]; hfaa[idx].y = 0.0;
        hfab[idx].x = tab[idx]; hfab[idx].y = 0.0;
        hfac[idx].x = tac[idx]; hfac[idx].y = 0.0;
        hfba[idx].x = tba[idx]; hfba[idx].y = 0.0;
        hfbb[idx].x = tbb[idx]; hfbb[idx].y = 0.0;
        hfbc[idx].x = tbc[idx]; hfbc[idx].y = 0.0;
        hfca[idx].x = tca[idx]; hfca[idx].y = 0.0;
        hfcb[idx].x = tcb[idx]; hfcb[idx].y = 0.0;
        hfcc[idx].x = tcc[idx]; hfcc[idx].y = 0.0;
    }
    free(taa);free(tab);free(tac);free(tba);free(tbb);
    free(tbc);free(tca);free(tcb);free(tcc);

    /* step 3: cuFFT Z2Z plan */
    {
        cufftResult r = cufftPlan2d(&d->plan, ny, nx, CUFFT_Z2Z);
        if (r != CUFFT_SUCCESS) {
            fprintf(stderr,"[demag] cufftPlan2d failed: %d\n",(int)r);
            free(hfaa);free(hfab);free(hfac);free(hfba);free(hfbb);
            free(hfbc);free(hfca);free(hfcb);free(hfcc);free(d);
            return NULL;
        }
    }

    /* step 4: H2D + FFT tensor */
    cufftDoubleComplex *difaa=NULL,*difab=NULL,*difac=NULL;
    cufftDoubleComplex *difba=NULL,*difbb=NULL,*difbc=NULL;
    cufftDoubleComplex *difca=NULL,*difcb=NULL,*difcc=NULL;
    cufftDoubleComplex *dofaa=NULL,*dofab=NULL,*dofac=NULL;
    cufftDoubleComplex *dofba=NULL,*dofbb=NULL,*dofbc=NULL;
    cufftDoubleComplex *dofca=NULL,*dofcb=NULL,*dofcc=NULL;

    cudaMalloc((void**)&difaa,csz); cudaMalloc((void**)&dofaa,csz);
    cudaMalloc((void**)&difab,csz); cudaMalloc((void**)&dofab,csz);
    cudaMalloc((void**)&difac,csz); cudaMalloc((void**)&dofac,csz);
    cudaMalloc((void**)&difba,csz); cudaMalloc((void**)&dofba,csz);
    cudaMalloc((void**)&difbb,csz); cudaMalloc((void**)&dofbb,csz);
    cudaMalloc((void**)&difbc,csz); cudaMalloc((void**)&dofbc,csz);
    cudaMalloc((void**)&difca,csz); cudaMalloc((void**)&dofca,csz);
    cudaMalloc((void**)&difcb,csz); cudaMalloc((void**)&dofcb,csz);
    cudaMalloc((void**)&difcc,csz); cudaMalloc((void**)&dofcc,csz);

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

    cufftExecZ2Z(d->plan,difaa,dofaa,CUFFT_FORWARD);
    cufftExecZ2Z(d->plan,difab,dofab,CUFFT_FORWARD);
    cufftExecZ2Z(d->plan,difac,dofac,CUFFT_FORWARD);
    cufftExecZ2Z(d->plan,difba,dofba,CUFFT_FORWARD);
    cufftExecZ2Z(d->plan,difbb,dofbb,CUFFT_FORWARD);
    cufftExecZ2Z(d->plan,difbc,dofbc,CUFFT_FORWARD);
    cufftExecZ2Z(d->plan,difca,dofca,CUFFT_FORWARD);
    cufftExecZ2Z(d->plan,difcb,dofcb,CUFFT_FORWARD);
    cufftExecZ2Z(d->plan,difcc,dofcc,CUFFT_FORWARD);

    /* step 5: D2H → hofaa..hofcc (keep on host for multiply loop) */
    d->hofaa=(cufftDoubleComplex*)malloc(csz); d->hofab=(cufftDoubleComplex*)malloc(csz);
    d->hofac=(cufftDoubleComplex*)malloc(csz); d->hofba=(cufftDoubleComplex*)malloc(csz);
    d->hofbb=(cufftDoubleComplex*)malloc(csz); d->hofbc=(cufftDoubleComplex*)malloc(csz);
    d->hofca=(cufftDoubleComplex*)malloc(csz); d->hofcb=(cufftDoubleComplex*)malloc(csz);
    d->hofcc=(cufftDoubleComplex*)malloc(csz);

    cudaMemcpy(d->hofaa,dofaa,csz,cudaMemcpyDeviceToHost);
    cudaMemcpy(d->hofab,dofab,csz,cudaMemcpyDeviceToHost);
    cudaMemcpy(d->hofac,dofac,csz,cudaMemcpyDeviceToHost);
    cudaMemcpy(d->hofba,dofba,csz,cudaMemcpyDeviceToHost);
    cudaMemcpy(d->hofbb,dofbb,csz,cudaMemcpyDeviceToHost);
    cudaMemcpy(d->hofbc,dofbc,csz,cudaMemcpyDeviceToHost);
    cudaMemcpy(d->hofca,dofca,csz,cudaMemcpyDeviceToHost);
    cudaMemcpy(d->hofcb,dofcb,csz,cudaMemcpyDeviceToHost);
    cudaMemcpy(d->hofcc,dofcc,csz,cudaMemcpyDeviceToHost);

    cudaFree(difaa);cudaFree(difab);cudaFree(difac);
    cudaFree(difba);cudaFree(difbb);cudaFree(difbc);
    cudaFree(difca);cudaFree(difcb);cudaFree(difcc);
    cudaFree(dofaa);cudaFree(dofab);cudaFree(dofac);
    cudaFree(dofba);cudaFree(dofbb);cudaFree(dofbc);
    cudaFree(dofca);cudaFree(dofcb);cudaFree(dofcc);

    /* allocate persistent per-timestep buffers */
    cudaMalloc((void**)&d->dima,csz); cudaMalloc((void**)&d->doma,csz);
    cudaMalloc((void**)&d->dimb,csz); cudaMalloc((void**)&d->domb,csz);
    cudaMalloc((void**)&d->dimc,csz); cudaMalloc((void**)&d->domc,csz);
    cudaMalloc((void**)&d->dkha,csz); cudaMalloc((void**)&d->drha,csz);
    cudaMalloc((void**)&d->dkhb,csz); cudaMalloc((void**)&d->drhb,csz);
    cudaMalloc((void**)&d->dkhc,csz); cudaMalloc((void**)&d->drhc,csz);

    d->homa=(cufftDoubleComplex*)malloc(csz); d->homb=(cufftDoubleComplex*)malloc(csz);
    d->homc=(cufftDoubleComplex*)malloc(csz); d->hkha=(cufftDoubleComplex*)malloc(csz);
    d->hkhb=(cufftDoubleComplex*)malloc(csz); d->hkhc=(cufftDoubleComplex*)malloc(csz);
    d->hrha=(cufftDoubleComplex*)malloc(csz); d->hrhb=(cufftDoubleComplex*)malloc(csz);
    d->hrhc=(cufftDoubleComplex*)malloc(csz);

    printf("[Demag] Ready. Host ~%.1f MB  Device ~%.1f MB\n",
           18.0*csz/1e6, 12.0*csz/1e6);
    return d;
}

/* 
 * Demag_Apply  (steps 6-14, per-timestep)
 *  */
void Demag_Apply(DemagData *d, const double *y_dev, double *h_out)
{
    if (!d) return;

    const int nx    = d->nx;
    const int ny    = d->ny;
    const int ncell = d->ncell;
    const int nn2   = d->nn2;
    const double scale = d->strength / (double)(nx * ny);
    const size_t csz = (size_t)ncell * sizeof(cufftDoubleComplex);

    /* step 6: D2H y, pack into complex hma_c..hmc_c */
    double *h_y = (double*)malloc((size_t)3 * ncell * sizeof(double));
    if (!h_y) { fprintf(stderr,"[demag] h_y alloc failed\n"); return; }
    cudaMemcpy(h_y, y_dev, (size_t)3*ncell*sizeof(double), cudaMemcpyDeviceToHost);

    cufftDoubleComplex *hma_c=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hmb_c=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hmc_c=(cufftDoubleComplex*)malloc(csz);

    for (int idx = 0; idx < ncell; idx++) {
        hma_c[idx].x = h_y[idx];           hma_c[idx].y = 0.0;
        hmb_c[idx].x = h_y[ncell + idx];   hmb_c[idx].y = 0.0;
        hmc_c[idx].x = h_y[2*ncell + idx]; hmc_c[idx].y = 0.0;
    }
    free(h_y);

    /* step 7: H2D */
    cudaMemcpy(d->dima, hma_c, csz, cudaMemcpyHostToDevice);
    cudaMemcpy(d->dimb, hmb_c, csz, cudaMemcpyHostToDevice);
    cudaMemcpy(d->dimc, hmc_c, csz, cudaMemcpyHostToDevice);
    free(hma_c); free(hmb_c); free(hmc_c);

    /* step 8: FFT FORWARD */
    cufftExecZ2Z(d->plan, d->dima, d->doma, CUFFT_FORWARD);
    cufftExecZ2Z(d->plan, d->dimb, d->domb, CUFFT_FORWARD);
    cufftExecZ2Z(d->plan, d->dimc, d->domc, CUFFT_FORWARD);

    /* step 9: D2H */
    cudaMemcpy(d->homa, d->doma, csz, cudaMemcpyDeviceToHost);
    cudaMemcpy(d->homb, d->domb, csz, cudaMemcpyDeviceToHost);
    cudaMemcpy(d->homc, d->domc, csz, cudaMemcpyDeviceToHost);

    /* step 10: host multiply — EXACT copy of  cadd/cmul loop */
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int idx = j * nx + i;
            d->hkha[idx] = h_cadd(h_cadd(h_cmul(d->hofaa[idx], d->homa[idx]),
                                          h_cmul(d->hofab[idx], d->homb[idx])),
                                          h_cmul(d->hofac[idx], d->homc[idx]));
            d->hkhb[idx] = h_cadd(h_cadd(h_cmul(d->hofba[idx], d->homa[idx]),
                                          h_cmul(d->hofbb[idx], d->homb[idx])),
                                          h_cmul(d->hofbc[idx], d->homc[idx]));
            d->hkhc[idx] = h_cadd(h_cadd(h_cmul(d->hofca[idx], d->homa[idx]),
                                          h_cmul(d->hofcb[idx], d->homb[idx])),
                                          h_cmul(d->hofcc[idx], d->homc[idx]));
        }
    }

    /* step 11: H2D */
    cudaMemcpy(d->dkha, d->hkha, csz, cudaMemcpyHostToDevice);
    cudaMemcpy(d->dkhb, d->hkhb, csz, cudaMemcpyHostToDevice);
    cudaMemcpy(d->dkhc, d->hkhc, csz, cudaMemcpyHostToDevice);

    /* step 12: FFT INVERSE */
    cufftExecZ2Z(d->plan, d->dkha, d->drha, CUFFT_INVERSE);
    cufftExecZ2Z(d->plan, d->dkhb, d->drhb, CUFFT_INVERSE);
    cufftExecZ2Z(d->plan, d->dkhc, d->drhc, CUFFT_INVERSE);

    /* step 13: D2H */
    cudaMemcpy(d->hrha, d->drha, csz, cudaMemcpyDeviceToHost);
    cudaMemcpy(d->hrhb, d->drhb, csz, cudaMemcpyDeviceToHost);
    cudaMemcpy(d->hrhc, d->drhc, csz, cudaMemcpyDeviceToHost);

    /* step 14: scatter into SoA h_out with  index remapping
     *
     *  output loop (from main()):
     *   for j: jy = (j < nn2) ? (nn2-j) : (nn-j+nn2-1)
     *   for i: ix = (i < nn2) ? (nn2-i) : (nn-i+nn2-1)
     *   idxnew = jy*nn + ix
     *   hx = hrha[idxnew].x / (nn*nn)
     *
     * h_out is device memory — read existing values, add, write back.
     */
    double *h_hout = (double*)malloc((size_t)3*ncell*sizeof(double));
    if (!h_hout) { fprintf(stderr,"[demag] h_hout alloc failed\n"); return; }
    cudaMemcpy(h_hout, h_out, (size_t)3*ncell*sizeof(double), cudaMemcpyDeviceToHost);

    for (int j = 0; j < ny; j++) {
        int jy = (j < nn2) ? (nn2 - j) : (ny - j + nn2 - 1);
        for (int i = 0; i < nx; i++) {
            int ix = (i < nn2) ? (nn2 - i) : (nx - i + nn2 - 1);
            int idx    = j  * nx + i;
            int idxnew = jy * nx + ix;
            if (idxnew < 0 || idxnew >= ncell) continue;

            h_hout[idx]          += d->hrha[idxnew].x * scale;
            h_hout[ncell + idx]  += d->hrhb[idxnew].x * scale;
            h_hout[2*ncell + idx]+= d->hrhc[idxnew].x * scale;
        }
    }
    cudaMemcpy(h_out, h_hout, (size_t)3*ncell*sizeof(double), cudaMemcpyHostToDevice);
    free(h_hout);
}

/* 
 * Demag_Destroy
 *  */
void Demag_Destroy(DemagData *d)
{
    if (!d) return;
    cufftDestroy(d->plan);
    free(d->hofaa); free(d->hofab); free(d->hofac);
    free(d->hofba); free(d->hofbb); free(d->hofbc);
    free(d->hofca); free(d->hofcb); free(d->hofcc);
    cudaFree(d->dima); cudaFree(d->doma);
    cudaFree(d->dimb); cudaFree(d->domb);
    cudaFree(d->dimc); cudaFree(d->domc);
    cudaFree(d->dkha); cudaFree(d->drha);
    cudaFree(d->dkhb); cudaFree(d->drhb);
    cudaFree(d->dkhc); cudaFree(d->drhc);
    free(d->homa); free(d->homb); free(d->homc);
    free(d->hkha); free(d->hkhb); free(d->hkhc);
    free(d->hrha); free(d->hrhb); free(d->hrhc);
    free(d);
}
