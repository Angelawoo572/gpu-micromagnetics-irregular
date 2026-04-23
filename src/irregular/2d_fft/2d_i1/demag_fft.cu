/*
 * demag_fft_test.cu
 *
 * Standalone test for the GPU-only FFT demag pipeline.
 * Goal:
 *   - use the SAME magnetization pattern as demag_test.cu
 *   - print the SAME output format as demag_test.cu
 *   - compare numerically against demag_test
 *
 * Compile:
 *   nvcc -O3 -arch=sm_89 demag_fft_test.cu -lcufft -o demag_fft_test
 *
 * Run:
 *   ./demag_test      > out_v1.txt
 *   ./demag_fft_test  > out_v2.txt
 *   diff out_v1.txt out_v2.txt
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cufft.h>
#include <cuda_runtime.h>

/* =========================
   Problem size / constants
   ========================= */
static const int NN = 64;
static const double THICK = 100.0;
static const double STRENGTH = 1.0;

/* =========================
   Newell tensor helpers
   ========================= */

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

static void calt(double thik, int mdx, int mdy,
                 double taa[], double tab[], double tac[],
                 double tba[], double tbb[], double tbc[],
                 double tca[], double tcb[], double tcc[])
{
    int mdx2=mdx/2, mdy2=mdy/2;
    double a=0.49999, b=0.5*thik;
    double dm[10];

    for (int j=0;j<mdy;j++) {
        for (int i=0;i<mdx;i++) {
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
}

/* =========================
   GPU kernels
   ========================= */

__global__ static void pack_m_kernel(
    const double*            __restrict__ y_dev,
    cufftDoubleComplex*      __restrict__ dima,
    cufftDoubleComplex*      __restrict__ dimb,
    cufftDoubleComplex*      __restrict__ dimc,
    int ncell)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncell) return;
    dima[idx].x = y_dev[idx];
    dima[idx].y = 0.0;
    dimb[idx].x = y_dev[ncell + idx];
    dimb[idx].y = 0.0;
    dimc[idx].x = y_dev[2*ncell + idx];
    dimc[idx].y = 0.0;
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

/* IMPORTANT:
 * demag_test uses the old reflection-style remap:
 *   jy = (j<nn2) ? (nn2-j) : (nn-j+nn2-1)
 *   ix = (i<nn2) ? (nn2-i) : (nn-i+nn2-1)
 * We reproduce THAT exactly here so the printed output matches.
 */
__global__ static void remap_to_soa_same_as_demag_test_kernel(
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

    int dst = j * nx + i;

    int jy = (j < ny2) ? (ny2 - j) : (ny - j + ny2 - 1);
    int ix = (i < nx2) ? (nx2 - i) : (nx - i + nx2 - 1);
    int src = jy * nx + ix;

    h_out[dst]           = drha[src].x * scale;
    h_out[ncell + dst]   = drhb[src].x * scale;
    h_out[2*ncell + dst] = drhc[src].x * scale;
}

/* =========================
   Main test
   ========================= */

int main()
{
    const int nx = NN;
    const int ny = NN;
    const int ncell = nx * ny;
    const int nx2 = nx / 2;
    const int ny2 = ny / 2;
    const size_t csz = (size_t)ncell * sizeof(cufftDoubleComplex);
    const size_t rsz = (size_t)(3 * ncell) * sizeof(double);

    /* ---- 1) Build tensor on CPU ---- */
    double *taa=(double*)calloc(ncell,sizeof(double));
    double *tab=(double*)calloc(ncell,sizeof(double));
    double *tac=(double*)calloc(ncell,sizeof(double));
    double *tba=(double*)calloc(ncell,sizeof(double));
    double *tbb=(double*)calloc(ncell,sizeof(double));
    double *tbc=(double*)calloc(ncell,sizeof(double));
    double *tca=(double*)calloc(ncell,sizeof(double));
    double *tcb=(double*)calloc(ncell,sizeof(double));
    double *tcc=(double*)calloc(ncell,sizeof(double));

    calt(THICK, nx, ny, taa,tab,tac, tba,tbb,tbc, tca,tcb,tcc);

    /* ---- 2) Pack tensor into complex host arrays ---- */
    cufftDoubleComplex *hfaa=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfab=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfac=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfba=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfbb=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfbc=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfca=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfcb=(cufftDoubleComplex*)malloc(csz);
    cufftDoubleComplex *hfcc=(cufftDoubleComplex*)malloc(csz);

    for (int idx=0; idx<ncell; idx++) {
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

    free(taa); free(tab); free(tac);
    free(tba); free(tbb); free(tbc);
    free(tca); free(tcb); free(tcc);

    /* ---- 3) cuFFT plan ---- */
    cufftHandle plan;
    cufftPlan2d(&plan, ny, nx, CUFFT_Z2Z);

    /* ---- 4) Allocate device tensor arrays ---- */
    cufftDoubleComplex *difaa,*difab,*difac,*difba,*difbb,*difbc,*difca,*difcb,*difcc;
    cufftDoubleComplex *dofaa,*dofab,*dofac,*dofba,*dofbb,*dofbc,*dofca,*dofcb,*dofcc;

    cudaMalloc((void**)&difaa, csz); cudaMalloc((void**)&difab, csz); cudaMalloc((void**)&difac, csz);
    cudaMalloc((void**)&difba, csz); cudaMalloc((void**)&difbb, csz); cudaMalloc((void**)&difbc, csz);
    cudaMalloc((void**)&difca, csz); cudaMalloc((void**)&difcb, csz); cudaMalloc((void**)&difcc, csz);

    cudaMalloc((void**)&dofaa, csz); cudaMalloc((void**)&dofab, csz); cudaMalloc((void**)&dofac, csz);
    cudaMalloc((void**)&dofba, csz); cudaMalloc((void**)&dofbb, csz); cudaMalloc((void**)&dofbc, csz);
    cudaMalloc((void**)&dofca, csz); cudaMalloc((void**)&dofcb, csz); cudaMalloc((void**)&dofcc, csz);

    cudaMemcpy(difaa,hfaa,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difab,hfab,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difac,hfac,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difba,hfba,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difbb,hfbb,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difbc,hfbc,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difca,hfca,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difcb,hfcb,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difcc,hfcc,csz,cudaMemcpyHostToDevice);

    free(hfaa); free(hfab); free(hfac);
    free(hfba); free(hfbb); free(hfbc);
    free(hfca); free(hfcb); free(hfcc);

    cufftExecZ2Z(plan, difaa, dofaa, CUFFT_FORWARD);
    cufftExecZ2Z(plan, difab, dofab, CUFFT_FORWARD);
    cufftExecZ2Z(plan, difac, dofac, CUFFT_FORWARD);
    cufftExecZ2Z(plan, difba, dofba, CUFFT_FORWARD);
    cufftExecZ2Z(plan, difbb, dofbb, CUFFT_FORWARD);
    cufftExecZ2Z(plan, difbc, dofbc, CUFFT_FORWARD);
    cufftExecZ2Z(plan, difca, dofca, CUFFT_FORWARD);
    cufftExecZ2Z(plan, difcb, dofcb, CUFFT_FORWARD);
    cufftExecZ2Z(plan, difcc, dofcc, CUFFT_FORWARD);

    cudaFree(difaa); cudaFree(difab); cudaFree(difac);
    cudaFree(difba); cudaFree(difbb); cudaFree(difbc);
    cudaFree(difca); cudaFree(difcb); cudaFree(difcc);

    /* ---- 5) Build SAME magnetization pattern as demag_test ---- */
    double *h_y = (double*)calloc(3 * ncell, sizeof(double));

    int i1 = nx / 3, i2 = 2 * nx / 3;
    int j1 = ny / 3, j2 = 2 * ny / 3;

    for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
            int idx = j * nx + i;
            double ma = 1.0;
            if (i > i1 && i < i2 && j > j1 && j < j2) ma = 0.0;

            h_y[idx] = ma;            /* mx */
            h_y[ncell + idx] = 0.0;   /* my */
            h_y[2*ncell + idx] = 0.0; /* mz */
        }
    }

    /* ---- 6) Allocate device magnetization / workspace ---- */
    double *y_dev, *h_out_dev;
    cudaMalloc((void**)&y_dev, rsz);
    cudaMalloc((void**)&h_out_dev, rsz);
    cudaMemcpy(y_dev, h_y, rsz, cudaMemcpyHostToDevice);
    free(h_y);

    cufftDoubleComplex *dima,*dimb,*dimc,*doma,*domb,*domc;
    cufftDoubleComplex *dkha,*dkhb,*dkhc,*drha,*drhb,*drhc;

    cudaMalloc((void**)&dima, csz); cudaMalloc((void**)&doma, csz);
    cudaMalloc((void**)&dimb, csz); cudaMalloc((void**)&domb, csz);
    cudaMalloc((void**)&dimc, csz); cudaMalloc((void**)&domc, csz);

    cudaMalloc((void**)&dkha, csz); cudaMalloc((void**)&drha, csz);
    cudaMalloc((void**)&dkhb, csz); cudaMalloc((void**)&drhb, csz);
    cudaMalloc((void**)&dkhc, csz); cudaMalloc((void**)&drhc, csz);

    /* ---- 7) Pack y -> complex ---- */
    {
        int b = 256;
        int g = (ncell + b - 1) / b;
        pack_m_kernel<<<g, b>>>(y_dev, dima, dimb, dimc, ncell);
    }

    /* ---- 8) FFT M ---- */
    cufftExecZ2Z(plan, dima, doma, CUFFT_FORWARD);
    cufftExecZ2Z(plan, dimb, domb, CUFFT_FORWARD);
    cufftExecZ2Z(plan, dimc, domc, CUFFT_FORWARD);

    /* ---- 9) Multiply ---- */
    {
        int b = 256;
        int g = (ncell + b - 1) / b;
        multiply_kernel<<<g, b>>>(
            dofaa, dofab, dofac,
            dofba, dofbb, dofbc,
            dofca, dofcb, dofcc,
            doma, domb, domc,
            dkha, dkhb, dkhc,
            ncell);
    }

    /* ---- 10) IFFT H ---- */
    cufftExecZ2Z(plan, dkha, drha, CUFFT_INVERSE);
    cufftExecZ2Z(plan, dkhb, drhb, CUFFT_INVERSE);
    cufftExecZ2Z(plan, dkhc, drhc, CUFFT_INVERSE);

    /* ---- 11) Remap exactly like demag_test print path ---- */
    {
        dim3 block(16, 16);
        dim3 grid((nx + block.x - 1) / block.x,
                  (ny + block.y - 1) / block.y);
        double scale = STRENGTH / (double)(nx * ny);

        remap_to_soa_same_as_demag_test_kernel<<<grid, block>>>(
            drha, drhb, drhc,
            h_out_dev,
            scale,
            nx, ny, ncell, nx2, ny2);
    }

    /* ---- 12) Copy back and print SAME format as demag_test ---- */
    double *h_out = (double*)malloc(rsz);
    cudaMemcpy(h_out, h_out_dev, rsz, cudaMemcpyDeviceToHost);

    std::printf("%d %d %d\n", nx, ny, nx);
    for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
            int idx = j * nx + i;
            double hx = h_out[idx];
            double hy = h_out[ncell + idx];
            double hz = h_out[2*ncell + idx];
            std::printf("%f %f %f\n", hx, hy, hz);
        }
    }

    /* ---- cleanup ---- */
    free(h_out);

    cufftDestroy(plan);

    cudaFree(dofaa); cudaFree(dofab); cudaFree(dofac);
    cudaFree(dofba); cudaFree(dofbb); cudaFree(dofbc);
    cudaFree(dofca); cudaFree(dofcb); cudaFree(dofcc);

    cudaFree(dima); cudaFree(doma);
    cudaFree(dimb); cudaFree(domb);
    cudaFree(dimc); cudaFree(domc);

    cudaFree(dkha); cudaFree(drha);
    cudaFree(dkhb); cudaFree(drhb);
    cudaFree(dkhc); cudaFree(drhc);

    cudaFree(y_dev);
    cudaFree(h_out_dev);

    return 0;
}