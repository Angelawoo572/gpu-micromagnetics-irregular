/*
 * demag_direct_apply.cu
 *
 * Same Demag_Init/Apply/Destroy interface as demag_fft.cu.
 * Drop-in replacement for timing comparison:
 *
 *   FFT version:    h(i,j) = IFFT[ f̂aa · m̂ ]          O(N log N)
 *   Direct version: h(i,j) = Σ_{m,n} faa(i-m,j-n)·M(m,n)  O(N²)
 *
 *   faa, fab, ...  are the real-space demag tensor components (from calt/ctt)
 *   → h_x = Σ faa · m_x + Σ fab · m_y + Σ fac · m_z (direct conv)
 *   → ĥ_x = f̂aa · m̂_x + f̂ab · m̂_y + f̂ac · m̂_z (FFT conv)
 *
 * This file implements the left arrow (direct), demag_fft.cu implements
 * the right arrow (FFT). Everything else (calt/ctt, LLG integration,
 * CVODE setup) is identical.
 *
 * Compile into 2d_direct target via Makefile:
 *   DEMAG_SRC = demag_direct_apply.cu
 *   TARGET    = 2d_direct
 */

#include "demag_fft.h"   /* same header — same interface */

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
    double dnnn=sqrt(xn2+yn2+zn2), dpnn=sqrt(xp2+yn2+zn2);
    double dnpn=sqrt(xn2+yp2+zn2), dnnp=sqrt(xn2+yn2+zp2);
    double dppn=sqrt(xp2+yp2+zn2), dnpp=sqrt(xn2+yp2+zp2);
    double dpnp=sqrt(xp2+yn2+zp2), dppp=sqrt(xp2+yp2+zp2);

    dm[1] = atan(zn*yn/(xn*dnnn))-atan(zp*yn/(xn*dnnp))
           -atan(zn*yp/(xn*dnpn))+atan(zp*yp/(xn*dnpp))
           -atan(zn*yn/(xp*dpnn))+atan(zp*yn/(xp*dpnp))
           +atan(zn*yp/(xp*dppn))-atan(zp*yp/(xp*dppp));
    dm[2] = log((dnnn-zn)/(dnnp-zp))-log((dpnn-zn)/(dpnp-zp))
           -log((dnpn-zn)/(dnpp-zp))+log((dppn-zn)/(dppp-zp));
    dm[3] = log((dnnn-yn)/(dnpn-yp))-log((dpnn-yn)/(dppn-yp))
           -log((dnnp-yn)/(dnpp-yp))+log((dpnp-yn)/(dppp-yp));
    dm[4] = log((dnnn-zn)/(dnnp-zp))-log((dnpn-zn)/(dnpp-zp))
           -log((dpnn-zn)/(dpnp-zp))+log((dppn-zn)/(dppp-zp));
    dm[5] = atan(zn*xn/(yn*dnnn))-atan(zp*xn/(yn*dnnp))
           -atan(zn*xp/(yn*dpnn))+atan(zp*xp/(yn*dpnp))
           -atan(zn*xn/(yp*dnpn))+atan(zp*xn/(yp*dnpp))
           +atan(zn*xp/(yp*dppn))-atan(zp*xp/(yp*dppp));
    dm[6] = log((dnnn-xn)/(dpnn-xp))-log((dnpn-xn)/(dppn-xp))
           -log((dnnp-xn)/(dpnp-xp))+log((dnpp-xn)/(dppp-xp));
    dm[7] = log((dnnn-yn)/(dnpn-yp))-log((dnnp-yn)/(dnpp-yp))
           -log((dpnn-yn)/(dppn-yp))+log((dpnp-yn)/(dppp-yp));
    dm[8] = log((dnnn-xn)/(dpnn-xp))-log((dnnp-xn)/(dpnp-xp))
           -log((dnpn-xn)/(dppn-xp))+log((dnpp-xn)/(dppp-xp));
    dm[9] = atan(xn*yn/(zn*dnnn))-atan(xp*yn/(zn*dpnn))
           -atan(xn*yp/(zn*dnpn))+atan(xp*yp/(zn*dppn))
           -atan(xn*yn/(zp*dnnp))+atan(xp*yn/(zp*dpnp))
           +atan(xn*yp/(zp*dnpp))-atan(xp*yp/(zp*dppp));
}

/* ── calt: (1-indexed dm[]) ── */
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

/* 
 * CUDA kernel: O(N²) direct convolution
 *
 * Each thread = one target cell (ti, tj).
 * Loops over all source cells (si, sj):
 *
 *   h_x(ti,tj) = Σ_{si,sj} [ faa(di,dj)*mx(si,sj)
 *                            + fab(di,dj)*my(si,sj)
 *                            + fac(di,dj)*mz(si,sj) ]
 *
 * where (di,dj) = (ti-si, tj-sj) wrapped to tensor index.
 *
 * Tensor storage: faa[j*nx + i] with origin at (nx/2, ny/2),
 * so displacement (di, dj) → tensor index
 *   tx = ((di + nx/2) % nx + nx) % nx
 *   ty = ((dj + ny/2) % ny + ny) % ny
 *
 * This matches how calt stores values: faa[j*nx+i] for displacement
 * (i - nx/2, j - ny/2).
 *  */
__global__ static void direct_conv_kernel(
    /* source magnetization on device (SoA: mx|my|mz) */
    const double* __restrict__ mx,
    const double* __restrict__ my,
    const double* __restrict__ mz,
    /* demag tensor on device (9 components, faa[j*nx+i]) */
    const double* __restrict__ faa, const double* __restrict__ fab,
    const double* __restrict__ fac,
    const double* __restrict__ fba, const double* __restrict__ fbb,
    const double* __restrict__ fbc,
    const double* __restrict__ fca, const double* __restrict__ fcb,
    const double* __restrict__ fcc,
    /* output field (SoA: hx|hy|hz) */
    double* __restrict__ hx,
    double* __restrict__ hy,
    double* __restrict__ hz,
    int nx, int ny)
{
    /* target cell */
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    int tj = blockIdx.y * blockDim.y + threadIdx.y;
    if (ti >= nx || tj >= ny) return;

    const int nx2 = nx / 2;
    const int ny2 = ny / 2;

    double sumx = 0.0, sumy = 0.0, sumz = 0.0;

    for (int sj = 0; sj < ny; sj++) {
        for (int si = 0; si < nx; si++) {
            int src = sj * nx + si;
            double vmx = mx[src], vmy = my[src], vmz = mz[src];

            /* displacement target → source, wrapped to tensor index */
            int di = ti - si;
            int dj = tj - sj;
            int tx = ((di + nx2) % nx + nx) % nx;
            int ty = ((dj + ny2) % ny + ny) % ny;
            int tidx = ty * nx + tx;

            sumx += faa[tidx]*vmx + fab[tidx]*vmy + fac[tidx]*vmz;
            sumy += fba[tidx]*vmx + fbb[tidx]*vmy + fbc[tidx]*vmz;
            sumz += fca[tidx]*vmx + fcb[tidx]*vmy + fcc[tidx]*vmz;
        }
    }

    int dst = tj * nx + ti;
    hx[dst] = sumx;
    hy[dst] = sumy;
    hz[dst] = sumz;
}

/* 
 * DemagData for direct version
 *  */
struct DemagData {
    int nx, ny, ncell;
    double strength;

    /* device: tensor (9 components) */
    double *d_faa, *d_fab, *d_fac;
    double *d_fba, *d_fbb, *d_fbc;
    double *d_fca, *d_fcb, *d_fcc;

    /* device: separated M components (SoA split for kernel) */
    double *d_mx, *d_my, *d_mz;

    /* device: output H components */
    double *d_hx, *d_hy, *d_hz;
};

/* 
 * Demag_Init
 * Computes calt tensor and uploads to device. No FFT.
 *  */
DemagData* Demag_Init(int nx, int ny, double thick, double demag_strength)
{
    printf("[Demag DIRECT] Init: nx=%d ny=%d thick=%.4f strength=%.4f\n",
           nx, ny, thick, demag_strength);
    printf("[Demag DIRECT] O(N²) direct convolution — no FFT\n");
    printf("[Demag DIRECT] N=%d  N²=%lld\n",
           nx*ny, (long long)(nx*ny)*(nx*ny));

    DemagData *d = (DemagData*)calloc(1, sizeof(DemagData));
    if (!d) { fprintf(stderr,"[demag_direct] calloc failed\n"); return NULL; }

    d->nx       = nx;
    d->ny       = ny;
    d->ncell    = nx * ny;
    d->strength = demag_strength;

    const size_t rsz = (size_t)(nx * ny) * sizeof(double);

    /* step 1: compute tensor on CPU */
    printf("[Demag DIRECT] Computing calt (81-pt averaging)...\n");

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
        fprintf(stderr,"[demag_direct] tensor alloc failed\n");
        free(taa);free(tab);free(tac);free(tba);free(tbb);
        free(tbc);free(tca);free(tcb);free(tcc);free(d);
        return NULL;
    }
    calt(thick, nx, ny, taa,tab,tac, tba,tbb,tbc, tca,tcb,tcc);
    printf("[Demag DIRECT] calt done.\n");

    /* step 2: upload tensor to device */
    cudaMalloc((void**)&d->d_faa, rsz); cudaMalloc((void**)&d->d_fab, rsz);
    cudaMalloc((void**)&d->d_fac, rsz); cudaMalloc((void**)&d->d_fba, rsz);
    cudaMalloc((void**)&d->d_fbb, rsz); cudaMalloc((void**)&d->d_fbc, rsz);
    cudaMalloc((void**)&d->d_fca, rsz); cudaMalloc((void**)&d->d_fcb, rsz);
    cudaMalloc((void**)&d->d_fcc, rsz);

    cudaMemcpy(d->d_faa,taa,rsz,cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_fab,tab,rsz,cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_fac,tac,rsz,cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_fba,tba,rsz,cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_fbb,tbb,rsz,cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_fbc,tbc,rsz,cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_fca,tca,rsz,cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_fcb,tcb,rsz,cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_fcc,tcc,rsz,cudaMemcpyHostToDevice);

    free(taa);free(tab);free(tac);free(tba);free(tbb);
    free(tbc);free(tca);free(tcb);free(tcc);

    /* step 3: allocate per-timestep M and H device buffers */
    cudaMalloc((void**)&d->d_mx, rsz);
    cudaMalloc((void**)&d->d_my, rsz);
    cudaMalloc((void**)&d->d_mz, rsz);
    cudaMalloc((void**)&d->d_hx, rsz);
    cudaMalloc((void**)&d->d_hy, rsz);
    cudaMalloc((void**)&d->d_hz, rsz);

    printf("[Demag DIRECT] Ready. Device memory: %.1f MB\n",
           15.0 * rsz / 1e6);
    return d;
}

/* 
 * Demag_Apply  —  O(N²) direct convolution
 *
 * h_x(i,j) = Σ_{m,n} faa(i-m, j-n) · mx(m,n) + ...
 *           = direct sum over all source cells (no FFT)
 *
 * Corresponds to professor's:
 *   → h_x = Σ faa · m   (left arrow on board, space domain)
 *
 * vs FFT version:
 *   → ĥ_x = f̂aa · m̂   (right arrow, frequency domain)
 *  */
void Demag_Apply(DemagData *d, const double *y_dev, double *h_out)
{
    if (!d) return;

    const int nx    = d->nx;
    const int ny    = d->ny;
    const int ncell = d->ncell;
    const size_t rsz = (size_t)ncell * sizeof(double);

    /* split SoA y_dev [mx|my|mz] → separate device arrays */
    cudaMemcpy(d->d_mx, y_dev,          rsz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d->d_my, y_dev + ncell,  rsz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d->d_mz, y_dev + 2*ncell,rsz, cudaMemcpyDeviceToDevice);

    /* zero output H buffers */
    cudaMemset(d->d_hx, 0, rsz);
    cudaMemset(d->d_hy, 0, rsz);
    cudaMemset(d->d_hz, 0, rsz);

    /* launch O(N²) kernel: each thread computes one target cell */
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);

    direct_conv_kernel<<<grid, block>>>(
        d->d_mx, d->d_my, d->d_mz,
        d->d_faa, d->d_fab, d->d_fac,
        d->d_fba, d->d_fbb, d->d_fbc,
        d->d_fca, d->d_fcb, d->d_fcc,
        d->d_hx, d->d_hy, d->d_hz,
        nx, ny);

    cudaDeviceSynchronize();  /* wait for kernel before scatter */

    /* scatter h_x/h_y/h_z into SoA h_out, scaled by strength
     * h_out is device memory, read existing values and add */
    double *h_hout = (double*)malloc((size_t)3 * ncell * sizeof(double));
    if (!h_hout) { fprintf(stderr,"[demag_direct] h_hout alloc failed\n"); return; }
    cudaMemcpy(h_hout, h_out, (size_t)3*ncell*sizeof(double), cudaMemcpyDeviceToHost);

    double *h_hx = (double*)malloc(rsz);
    double *h_hy = (double*)malloc(rsz);
    double *h_hz = (double*)malloc(rsz);
    cudaMemcpy(h_hx, d->d_hx, rsz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hy, d->d_hy, rsz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hz, d->d_hz, rsz, cudaMemcpyDeviceToHost);

    for (int idx = 0; idx < ncell; idx++) {
        h_hout[idx]          += d->strength * h_hx[idx];
        h_hout[ncell + idx]  += d->strength * h_hy[idx];
        h_hout[2*ncell + idx]+= d->strength * h_hz[idx];
    }

    cudaMemcpy(h_out, h_hout, (size_t)3*ncell*sizeof(double), cudaMemcpyHostToDevice);
    free(h_hout); free(h_hx); free(h_hy); free(h_hz);
}

/* 
 * Demag_Destroy
 *  */
void Demag_Destroy(DemagData *d)
{
    if (!d) return;
    cudaFree(d->d_faa); cudaFree(d->d_fab); cudaFree(d->d_fac);
    cudaFree(d->d_fba); cudaFree(d->d_fbb); cudaFree(d->d_fbc);
    cudaFree(d->d_fca); cudaFree(d->d_fcb); cudaFree(d->d_fcc);
    cudaFree(d->d_mx);  cudaFree(d->d_my);  cudaFree(d->d_mz);
    cudaFree(d->d_hx);  cudaFree(d->d_hy);  cudaFree(d->d_hz);
    free(d);
}
