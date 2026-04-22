/*
 * demag_fft.cu
 *
 * Key optimization over v1:
 *   f̂ (tensor spectra dofaa..dofcc) computed ONCE at Demag_Init,
 *   stored permanently on device — never touched again.
 *
 *   Per-timestep Demag_Apply:
 *     OLD v1: FFT(M) on GPU → D2H → CPU multiply loop → H2D → IFFT on GPU
 *     NEW v2: FFT(M) on GPU → GPU multiply_kernel → IFFT on GPU
 *     Eliminates 9 D2H + 9 H2D transfers per f() call.
 *
 * Flow:
 *   Demag_Init (once):
 *   1. calt/ctt  → taa..tcc  (CPU)
 *   2. pack → complex hfaa..hfcc  (imag=0, CPU)
 *   3. H2D → difaa..difcc
 *   4. cufftExecZ2Z FORWARD → dofaa..dofcc  (GPU, PERMANENT)
 *   5. free difaa..difcc  (no longer needed)
 *
 *   Demag_Apply (every f() call):
 *   6. pack M from SoA → complex hma_c..hmc_c  (CPU)
 *   7. H2D → dima..dimc
 *   8. cufftExecZ2Z FORWARD → doma..domc  (GPU)
 *   9. multiply_kernel: dkha = dofaa*doma + dofab*domb + dofac*domc  (GPU)
 *  10. cufftExecZ2Z INVERSE → drha..drhc  (GPU)
 *  11. D2H → hrha..hrhc  (CPU)
 *  12. scatter hrha[idxnew].x/(nn*nn) into SoA h_out with index remap  (CPU)
 *
 * index remapping (step 12) is identical to demag_test.cu:
 *   jy = (j < nn2) ? (nn2-j) : (ny-j+nn2-1)
 *   ix = (i < nn2) ? (nn2-i) : (nx-i+nn2-1)
 *   idxnew = jy*nx + ix
 */

#include "demag_fft.h"

#include <cufft.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ── ctt: (1-indexed dm[]) ── */
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

/* ── calt: (1-indexed dm[]) ── */
static int calt(double thik, int mdx, int mdy,
                double taa[], double tab[], double tac[],
                double tba[], double tbb[], double tbc[],
                double tca[], double tcb[], double tcc[])
{
    int i, j, ix, jy, ikn;
    int mdx2=mdx/2, mdy2=mdy/2;
    double sx, sy;
    double a=0.49999, b=0.5*thik;
    double dm[10];

    for (j=0;j<mdy;j++) for (i=0;i<mdx;i++) {
        ikn=j*mdx+i;
        taa[ikn]=tab[ikn]=tac[ikn]=0.0;
        tba[ikn]=tbb[ikn]=tbc[ikn]=0.0;
        tca[ikn]=tcb[ikn]=tcc[ikn]=0.0;
        for (jy=-4;jy<=4;jy++) {
            sy=double(j-mdy2)+0.1*double(jy);
            for (ix=-4;ix<=4;ix++) {
                sx=double(i-mdx2)+0.1*double(ix);
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
    return 1;
}

/* 
 * GPU kernel: pointwise multiply  ĥ = f̂ · m̂
 *
 * f̂ (dofaa..dofcc) lives permanently on device (computed once at Init).
 * m̂ (doma/domb/domc) is computed fresh each Apply call.
 * Both are already on device — no transfers needed.
 *  */
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
    cufftDoubleComplex* __restrict__ dkha,
    cufftDoubleComplex* __restrict__ dkhb,
    cufftDoubleComplex* __restrict__ dkhc,
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

/* 
 * DemagData
 *  */
struct DemagData {
    int nx, ny, ncell;
    int nn2;
    double strength;

    cufftHandle plan;

    /* Device: f̂ tensor spectra — computed once at Init, permanent */
    cufftDoubleComplex *dofaa, *dofab, *dofac;
    cufftDoubleComplex *dofba, *dofbb, *dofbc;
    cufftDoubleComplex *dofca, *dofcb, *dofcc;

    /* Device: M FFT input/output */
    cufftDoubleComplex *dima, *dimb, *dimc;
    cufftDoubleComplex *doma, *domb, *domc;

    /* Device: H̃ spectra and IFFT output */
    cufftDoubleComplex *dkha, *dkhb, *dkhc;
    cufftDoubleComplex *drha, *drhb, *drhc;

    /* Host: IFFT result (for index-remapped scatter into h_out) */
    cufftDoubleComplex *hrha, *hrhb, *hrhc;
};

/* 
 * Demag_Init
 *  */
DemagData* Demag_Init(int nx, int ny, double thick, double demag_strength)
{
    printf("[Demag v2] Init: nx=%d ny=%d thick=%.4f strength=%.4f\n",
           nx, ny, thick, demag_strength);
    printf("[Demag v2] f̂ computed once at init, stays on device permanently.\n");

    DemagData *d = (DemagData*)calloc(1, sizeof(DemagData));
    if (!d) { fprintf(stderr,"[demag] calloc failed\n"); return NULL; }

    d->nx       = nx;
    d->ny       = ny;
    d->ncell    = nx * ny;
    d->nn2      = ny / 2;
    d->strength = demag_strength;

    const size_t csz = (size_t)(nx * ny) * sizeof(cufftDoubleComplex);

    /* step 1: calt on CPU */
    printf("[Demag v2] Computing calt (81-pt averaging)...\n");

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
    printf("[Demag v2] calt done.\n");

    /* step 2: pack tensor into complex (imag=0) */
    cufftDoubleComplex *hfaa=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfab=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfac=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfba=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfbb=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfbc=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfca=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfcb=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfcc=(cufftDoubleComplex*)malloc(csz);

    for (int idx=0;idx<nx*ny;idx++) {
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

    /* step 4: allocate permanent f̂ device arrays */
    cudaMalloc((void**)&d->dofaa,csz); cudaMalloc((void**)&d->dofab,csz);
    cudaMalloc((void**)&d->dofac,csz); cudaMalloc((void**)&d->dofba,csz);
    cudaMalloc((void**)&d->dofbb,csz); cudaMalloc((void**)&d->dofbc,csz);
    cudaMalloc((void**)&d->dofca,csz); cudaMalloc((void**)&d->dofcb,csz);
    cudaMalloc((void**)&d->dofcc,csz);

    /* temporary input buffers for H2D (freed after FFT) */
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

    /* FFT → dofXX are now permanently on device */
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

    /* step 5: free temporary input buffers */
    cudaFree(difaa);cudaFree(difab);cudaFree(difac);
    cudaFree(difba);cudaFree(difbb);cudaFree(difbc);
    cudaFree(difca);cudaFree(difcb);cudaFree(difcc);

    /* allocate persistent per-timestep device buffers */
    cudaMalloc((void**)&d->dima,csz); cudaMalloc((void**)&d->doma,csz);
    cudaMalloc((void**)&d->dimb,csz); cudaMalloc((void**)&d->domb,csz);
    cudaMalloc((void**)&d->dimc,csz); cudaMalloc((void**)&d->domc,csz);
    cudaMalloc((void**)&d->dkha,csz); cudaMalloc((void**)&d->drha,csz);
    cudaMalloc((void**)&d->dkhb,csz); cudaMalloc((void**)&d->drhb,csz);
    cudaMalloc((void**)&d->dkhc,csz); cudaMalloc((void**)&d->drhc,csz);

    /* host buffers for IFFT result */
    d->hrha=(cufftDoubleComplex*)malloc(csz);
    d->hrhb=(cufftDoubleComplex*)malloc(csz);
    d->hrhc=(cufftDoubleComplex*)malloc(csz);

    printf("[Demag v2] Device: %.1f MB  Host: %.1f MB\n",
           (double)(9+6+6)*csz/1e6, (double)(3)*csz/1e6);
    printf("[Demag v2] Ready.\n");
    return d;
}

/* 
 * Demag_Apply  (per-timestep, steps 6-12)
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

    /* step 6: D2H y, pack M → complex hma_c..hmc_c */
    {
        double *h_y=(double*)malloc((size_t)3*ncell*sizeof(double));
        if (!h_y) { fprintf(stderr,"[demag] h_y alloc failed\n"); return; }
        cudaMemcpy(h_y, y_dev, (size_t)3*ncell*sizeof(double),
                   cudaMemcpyDeviceToHost);

        cufftDoubleComplex *hma_c=(cufftDoubleComplex*)malloc(csz);
        cufftDoubleComplex *hmb_c=(cufftDoubleComplex*)malloc(csz);
        cufftDoubleComplex *hmc_c=(cufftDoubleComplex*)malloc(csz);

        for (int idx=0;idx<ncell;idx++) {
            hma_c[idx].x=h_y[idx];           hma_c[idx].y=0.0;
            hmb_c[idx].x=h_y[ncell+idx];     hmb_c[idx].y=0.0;
            hmc_c[idx].x=h_y[2*ncell+idx];   hmc_c[idx].y=0.0;
        }
        free(h_y);

        /* step 7: H2D */
        cudaMemcpy(d->dima, hma_c, csz, cudaMemcpyHostToDevice);
        cudaMemcpy(d->dimb, hmb_c, csz, cudaMemcpyHostToDevice);
        cudaMemcpy(d->dimc, hmc_c, csz, cudaMemcpyHostToDevice);
        free(hma_c); free(hmb_c); free(hmc_c);
    }

    /* step 8: FFT FORWARD M → m̂ (stays on device) */
    cufftExecZ2Z(d->plan, d->dima, d->doma, CUFFT_FORWARD);
    cufftExecZ2Z(d->plan, d->dimb, d->domb, CUFFT_FORWARD);
    cufftExecZ2Z(d->plan, d->dimc, d->domc, CUFFT_FORWARD);

    /* step 9: GPU multiply  ĥ = f̂ · m̂  (all on device, no transfers) */
    {
        int block=256, grid=(ncell+block-1)/block;
        multiply_kernel<<<grid,block>>>(
            d->dofaa, d->dofab, d->dofac,
            d->dofba, d->dofbb, d->dofbc,
            d->dofca, d->dofcb, d->dofcc,
            d->doma,  d->domb,  d->domc,
            d->dkha,  d->dkhb,  d->dkhc,
            ncell);
    }

    /* step 10: FFT INVERSE ĥ → h */
    cufftExecZ2Z(d->plan, d->dkha, d->drha, CUFFT_INVERSE);
    cufftExecZ2Z(d->plan, d->dkhb, d->drhb, CUFFT_INVERSE);
    cufftExecZ2Z(d->plan, d->dkhc, d->drhc, CUFFT_INVERSE);

    /* step 11: D2H result */
    cudaMemcpy(d->hrha, d->drha, csz, cudaMemcpyDeviceToHost);
    cudaMemcpy(d->hrhb, d->drhb, csz, cudaMemcpyDeviceToHost);
    cudaMemcpy(d->hrhc, d->drhc, csz, cudaMemcpyDeviceToHost);

    /* step 12: scatter with  index remapping → h_out */
    double *h_hout=(double*)malloc((size_t)3*ncell*sizeof(double));
    if (!h_hout) { fprintf(stderr,"[demag] h_hout alloc failed\n"); return; }
    cudaMemcpy(h_hout, h_out, (size_t)3*ncell*sizeof(double),
               cudaMemcpyDeviceToHost);

    for (int j=0;j<ny;j++) {
        int jy = (j<nn2) ? (nn2-j) : (ny-j+nn2-1);
        for (int i=0;i<nx;i++) {
            int ix     = (i<nn2) ? (nn2-i) : (nx-i+nn2-1);
            int idx    = j*nx + i;
            int idxnew = jy*nx + ix;
            if (idxnew<0 || idxnew>=ncell) continue;

            h_hout[idx]          += d->hrha[idxnew].x * scale;
            h_hout[ncell+idx]    += d->hrhb[idxnew].x * scale;
            h_hout[2*ncell+idx]  += d->hrhc[idxnew].x * scale;
        }
    }

    cudaMemcpy(h_out, h_hout, (size_t)3*ncell*sizeof(double),
               cudaMemcpyHostToDevice);
    free(h_hout);
}

/* 
 * Demag_Destroy
 *  */
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

    free(d->hrha); free(d->hrhb); free(d->hrhc);
    free(d);
}
