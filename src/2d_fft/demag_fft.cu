/*
 * demag_fft.cu  —  FFT demagnetization field, professor's Newell tensor method
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * PHYSICS
 * ═══════════════════════════════════════════════════════════════════════════
 * The demagnetization field is the convolution:
 *
 *   h_dmag,α(i,j) = Σ_{m,n}  N_αβ(i-m, j-n) · M_β(m,n)
 *
 * Via the convolution theorem:
 *
 *   h_dmag,α = IFFT[ N̂_αβ(k) · M̂_β(k) ]
 *
 * N_αβ(r) is the demag tensor computed by professor's calt/ctt functions
 * (Newell et al. analytic integrals for a rectangular prism source cell).
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * PROFESSOR'S ALGORITHM (faithfully reproduced from pseudocode)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * ctt(b, a, sx, sy, dm[]):
 *   Computes the 9 demag tensor components for a displacement (sx,sy,0)
 *   using closed-form atan/log integrals for a prism of half-size (a,a,b).
 *   a = 0.49999  (half-cell size in x and y)
 *   b = 0.5*thick (half-cell size in z)
 *
 * calt(thick, nx, ny, taa..tcc):
 *   For each grid point (i,j), samples the ctt kernel on a 9×9 sub-grid
 *   (jx,jy = -4..4 in steps of 0.1*cell) and averages (divides by 81).
 *   This is a numerical integration to reduce aliasing of the singular
 *   demag kernel at short range.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * cuFFT USAGE
 * ═══════════════════════════════════════════════════════════════════════════
 * Uses C2C (Z2Z) plans, matching the professor's cufftPlan2d CUFFT_Z2Z.
 * The real-valued tensor components are zero-padded into complex arrays
 * (imaginary part = 0) before FFT, exactly as in the pseudocode.
 *
 * Plan:  cufftPlan2d(&plan, ny, nx, CUFFT_Z2Z)
 * Forward: cufftExecZ2Z(..., CUFFT_FORWARD)
 * Inverse: cufftExecZ2Z(..., CUFFT_INVERSE)  + normalize by 1/(nx*ny)
 */

#include "demag_fft.h"

#include <cufft.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nvector/nvector_cuda.h>

/* ─── internal macros ─────────────────────────────────────────────────── */
#define CHECK_CUDA(call) do {                                               \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr,"[demag] CUDA %s:%d: %s\n",                         \
                __FILE__,__LINE__,cudaGetErrorString(_e));                  \
        return NULL;                                                        \
    }                                                                      \
} while(0)

#define CHECK_CUDA_V(call) do {                                             \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess)                                                  \
        fprintf(stderr,"[demag] CUDA %s:%d: %s\n",                         \
                __FILE__,__LINE__,cudaGetErrorString(_e));                  \
} while(0)

#define CHECK_CUFFT(call) do {                                              \
    cufftResult _r = (call);                                                \
    if (_r != CUFFT_SUCCESS) {                                              \
        fprintf(stderr,"[demag] cuFFT %s:%d: code=%d\n",                   \
                __FILE__,__LINE__,(int)_r);                                 \
        return NULL;                                                        \
    }                                                                      \
} while(0)

#define CHECK_CUFFT_V(call) do {                                            \
    cufftResult _r = (call);                                                \
    if (_r != CUFFT_SUCCESS)                                                \
        fprintf(stderr,"[demag] cuFFT %s:%d: code=%d\n",                   \
                __FILE__,__LINE__,(int)_r);                                 \
} while(0)

/* safe log — avoids log(0) */
static inline double salog(double x) {
    return (x > 1e-300) ? log(x) : -300.0*log(10.0);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ctt  —  analytic demag tensor for one displacement (sx, sy)
 *
 * Directly translated from professor's Fortran-style pseudocode.
 * Computes dm[0..8] = [Nxx, Nxy, Nxz, Nyx, Nyy, Nyz, Nzx, Nzy, Nzz]
 * (0-indexed here, professor used 1-indexed)
 *
 * b = half-thickness in z  (= 0.5 * thick)
 * a = half-cell size in x,y (= 0.49999)
 * ═══════════════════════════════════════════════════════════════════════════ */
static void ctt(double b, double a, double sx, double sy, double dm[9])
{
    double sz = 0.0;

    double xn = sx - a,  xp = sx + a;
    double yn = sy - a,  yp = sy + a;
    double zn = sz - b,  zp = sz + b;

    double xn2 = xn*xn, xp2 = xp*xp;
    double yn2 = yn*yn, yp2 = yp*yp;
    double zn2 = zn*zn, zp2 = zp*zp;

    double dnnn = sqrt(xn2+yn2+zn2);
    double dpnn = sqrt(xp2+yn2+zn2);
    double dnpn = sqrt(xn2+yp2+zn2);
    double dnnp = sqrt(xn2+yn2+zp2);
    double dppn = sqrt(xp2+yp2+zn2);
    double dnpp = sqrt(xn2+yp2+zp2);
    double dpnp = sqrt(xp2+yn2+zp2);
    double dppp = sqrt(xp2+yp2+zp2);

    /* dm[0] = Nxx  — use atan(y/x), NOT atan2, to match professor's formula.
     * atan2(y,x) differs from atan(y/x) by ±π when x<0, corrupting cells
     * in the left half of the grid (i < nx/2, where xn = sx-a < 0). */
    dm[0] =  atan(zn*yn/(xn*dnnn)) - atan(zp*yn/(xn*dnnp))
            -atan(zn*yp/(xn*dnpn)) + atan(zp*yp/(xn*dnpp))
            -atan(zn*yn/(xp*dpnn)) + atan(zp*yn/(xp*dpnp))
            +atan(zn*yp/(xp*dppn)) - atan(zp*yp/(xp*dppp));

    /* dm[1] = Nxy */
    dm[1] =  salog(dnnn-zn) - salog(dnnp-zp)
            -salog(dpnn-zn) + salog(dpnp-zp)
            -salog(dnpn-zn) + salog(dnpp-zp)
            +salog(dppn-zn) - salog(dppp-zp);

    /* dm[2] = Nxz */
    dm[2] =  salog(dnnn-yn) - salog(dnpn-yp)
            -salog(dpnn-yn) + salog(dppn-yp)
            -salog(dnnp-yn) + salog(dnpp-yp)
            +salog(dpnp-yn) - salog(dppp-yp);

    /* dm[3] = Nyx  (= Nxy by symmetry, kept separate for clarity) */
    dm[3] =  salog(dnnn-zn) - salog(dnnp-zp)
            -salog(dnpn-zn) + salog(dnpp-zp)
            -salog(dpnn-zn) + salog(dpnp-zp)
            +salog(dppn-zn) - salog(dppp-zp);

    /* dm[4] = Nyy  — use atan(y/x), same reason as dm[0]. */
    dm[4] =  atan(zn*xn/(yn*dnnn)) - atan(zp*xn/(yn*dnnp))
            -atan(zn*xp/(yn*dpnn)) + atan(zp*xp/(yn*dpnp))
            -atan(zn*xn/(yp*dnpn)) + atan(zp*xn/(yp*dnpp))
            +atan(zn*xp/(yp*dppn)) - atan(zp*xp/(yp*dppp));

    /* dm[5] = Nyz */
    dm[5] =  salog(dnnn-xn) - salog(dpnn-xp)
            -salog(dnpn-xn) + salog(dppn-xp)
            -salog(dnnp-xn) + salog(dpnp-xp)
            +salog(dnpp-xn) - salog(dppp-xp);

    /* dm[6] = Nzx */
    dm[6] =  salog(dnnn-yn) - salog(dnpn-yp)
            -salog(dnnp-yn) + salog(dnpp-yp)
            -salog(dpnn-yn) + salog(dppn-yp)
            +salog(dpnp-yn) - salog(dppp-yp);

    /* dm[7] = Nzy */
    dm[7] =  salog(dnnn-xn) - salog(dpnn-xp)
            -salog(dnnp-xn) + salog(dpnp-xp)
            -salog(dnpn-xn) + salog(dppn-xp)
            +salog(dnpp-xn) - salog(dppp-xp);

    /* dm[8] = Nzz  — use atan(y/x), same reason as dm[0].
     * zn = -0.5*thick < 0 always, so atan2 would differ by π whenever
     * the numerator sign changes, producing wrong Nzz everywhere. */
    dm[8] =  atan(xn*yn/(zn*dnnn)) - atan(xp*yn/(zn*dpnn))
            -atan(xn*yp/(zn*dnpn)) + atan(xp*yp/(zn*dppn))
            -atan(xn*yn/(zp*dnnp)) + atan(xp*yn/(zp*dpnp))
            +atan(xn*yp/(zp*dnpp)) - atan(xp*yp/(zp*dppp));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * calt  —  compute demag tensor on the full nx×ny grid
 *
 * Faithfully implements professor's pseudocode:
 *   For each cell (i,j), sample ctt on a 9×9 sub-grid (ix,jy = -4..4)
 *   at offsets 0.1*ix, 0.1*jy — then average (divide by 81).
 *
 * The cell displacement is (i - nx/2) + 0.1*ix  etc., which centres
 * the origin at the middle of the grid (matching professor's ix-mdx2).
 *
 * Output arrays: taa..tcc [nx*ny doubles], laid out as taa[j*nx+i].
 * ═══════════════════════════════════════════════════════════════════════════ */
static void calt(double thick, int nx, int ny,
                 double *taa, double *tab, double *tac,
                 double *tba, double *tbb, double *tbc,
                 double *tca, double *tcb, double *tcc)
{
    const int nx2 = nx / 2;
    const int ny2 = ny / 2;
    const double a = 0.49999;          /* half-cell in x,y */
    const double b = 0.5 * thick;     /* half-cell in z   */
    double dm[9];

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int ikn = j * nx + i;

            /* zero all 9 components */
            taa[ikn] = tbb[ikn] = tcc[ikn] = 0.0;
            tab[ikn] = tac[ikn] = 0.0;
            tba[ikn] = tbc[ikn] = 0.0;
            tca[ikn] = tcb[ikn] = 0.0;

            /* 9×9 sub-grid averaging (professor's loop ix,jy = -4..4) */
            for (int jy = -4; jy <= 4; jy++) {
                double sy = (double)(j - ny2) + 0.1*(double)jy;
                for (int ix = -4; ix <= 4; ix++) {
                    double sx = (double)(i - nx2) + 0.1*(double)ix;

                    ctt(b, a, sx, sy, dm);

                    taa[ikn] += dm[0];
                    tab[ikn] += dm[1];
                    tac[ikn] += dm[2];
                    tba[ikn] += dm[3];
                    tbb[ikn] += dm[4];
                    tbc[ikn] += dm[5];
                    tca[ikn] += dm[6];
                    tcb[ikn] += dm[7];
                    tcc[ikn] += dm[8];
                }
            }

            /* average over 9*9 = 81 sub-samples */
            taa[ikn] /= 81.0;  tab[ikn] /= 81.0;  tac[ikn] /= 81.0;
            tba[ikn] /= 81.0;  tbb[ikn] /= 81.0;  tbc[ikn] /= 81.0;
            tca[ikn] /= 81.0;  tcb[ikn] /= 81.0;  tcc[ikn] /= 81.0;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GPU kernels
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * pointwise_multiply_kernel
 *
 * For each k-space point:
 *   Ĥ_α(k) = Σ_β  N̂_αβ(k) · M̂_β(k)
 *
 * N̂ arrays may be complex (result of FFT of real-valued tensor),
 * so we do full complex×complex multiply.
 *
 * Tensor index map (matches professor's dm[0..8]):
 *   N̂aa=xx, N̂ab=xy, N̂ac=xz
 *   N̂ba=yx, N̂bb=yy, N̂bc=yz
 *   N̂ca=zx, N̂cb=zy, N̂cc=zz
 */
__global__ static void pointwise_multiply_kernel(
    /* magnetization spectra */
    const cufftDoubleComplex* __restrict__ Mx,
    const cufftDoubleComplex* __restrict__ My,
    const cufftDoubleComplex* __restrict__ Mz,
    /* demag tensor spectra (9 components) */
    const cufftDoubleComplex* __restrict__ Naa,
    const cufftDoubleComplex* __restrict__ Nab,
    const cufftDoubleComplex* __restrict__ Nac,
    const cufftDoubleComplex* __restrict__ Nba,
    const cufftDoubleComplex* __restrict__ Nbb,
    const cufftDoubleComplex* __restrict__ Nbc,
    const cufftDoubleComplex* __restrict__ Nca,
    const cufftDoubleComplex* __restrict__ Ncb,
    const cufftDoubleComplex* __restrict__ Ncc,
    /* output field spectra */
    cufftDoubleComplex* __restrict__ Hx,
    cufftDoubleComplex* __restrict__ Hy,
    cufftDoubleComplex* __restrict__ Hz,
    int nk)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nk) return;

    /* helper: complex multiply (a+ib)*(c+id) = (ac-bd) + i(ad+bc) */
#define CMUL_R(A,B) ((A)[k].x*(B)[k].x - (A)[k].y*(B)[k].y)
#define CMUL_I(A,B) ((A)[k].x*(B)[k].y + (A)[k].y*(B)[k].x)

    /* Ĥx = Naa*Mx + Nab*My + Nac*Mz */
    Hx[k].x = CMUL_R(Naa,Mx) + CMUL_R(Nab,My) + CMUL_R(Nac,Mz);
    Hx[k].y = CMUL_I(Naa,Mx) + CMUL_I(Nab,My) + CMUL_I(Nac,Mz);

    /* Ĥy = Nba*Mx + Nbb*My + Nbc*Mz */
    Hy[k].x = CMUL_R(Nba,Mx) + CMUL_R(Nbb,My) + CMUL_R(Nbc,Mz);
    Hy[k].y = CMUL_I(Nba,Mx) + CMUL_I(Nbb,My) + CMUL_I(Nbc,Mz);

    /* Ĥz = Nca*Mx + Ncb*My + Ncc*Mz */
    Hz[k].x = CMUL_R(Nca,Mx) + CMUL_R(Ncb,My) + CMUL_R(Ncc,Mz);
    Hz[k].y = CMUL_I(Nca,Mx) + CMUL_I(Ncb,My) + CMUL_I(Ncc,Mz);

#undef CMUL_R
#undef CMUL_I
}

/*
 * gather_to_complex_kernel
 * Copy one SoA real component → complex device buffer (imag=0).
 * SoA: y_soa[comp*ncell + cell]
 */
__global__ static void gather_to_complex_kernel(
    const double* __restrict__ y_soa,
    cufftDoubleComplex* __restrict__ buf,
    int comp, int ncell)
{
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= ncell) return;
    buf[cell].x = y_soa[comp * ncell + cell];
    buf[cell].y = 0.0;
}

/*
 * scatter_add_real_kernel
 * Add the real part of a complex buffer (scaled) into a SoA field array.
 * h_soa[comp*ncell + cell] += scale * buf[cell].x
 * (After IFFT of a real-sourced signal the result should be real.)
 */
__global__ static void scatter_add_real_kernel(
    const cufftDoubleComplex* __restrict__ buf,
    double* __restrict__ h_soa,
    int comp, int ncell, double scale)
{
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= ncell) return;
    h_soa[comp * ncell + cell] += scale * buf[cell].x;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * DemagData struct
 * ═══════════════════════════════════════════════════════════════════════════ */
struct DemagData {
    int nx, ny, ncell, nk;   /* nk = nx*ny (C2C uses full spectrum) */
    double scale;             /* 1/(nx*ny) for IFFT normalization     */
    double strength;          /* demag_strength prefactor             */

    cufftHandle plan;         /* single Z2Z plan for both fwd and inv */

    /* Device: magnetization spectra (3 components) */
    cufftDoubleComplex *d_Mhat[3];

    /* Device: field spectra (3 components) */
    cufftDoubleComplex *d_Hhat[3];

    /* Device: demag tensor spectra (9 components, pre-FFT'd at init) */
    cufftDoubleComplex *d_Nhat[9];   /* aa,ab,ac,ba,bb,bc,ca,cb,cc */

    /* Device: complex scratch for M gather and H scatter */
    cufftDoubleComplex *d_buf;       /* ncell complex doubles */
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Demag_Init
 * ═══════════════════════════════════════════════════════════════════════════ */
DemagData* Demag_Init(int nx, int ny, double thick, double demag_strength)
{
    printf("[Demag] Initializing Newell tensor (calt/ctt), nx=%d ny=%d thick=%.4f\n",
           nx, ny, thick);

    DemagData *d = (DemagData*)calloc(1, sizeof(DemagData));
    if (!d) { fprintf(stderr,"[demag] calloc failed\n"); return NULL; }

    d->nx      = nx;
    d->ny      = ny;
    d->ncell   = nx * ny;
    d->nk      = nx * ny;       /* Z2Z: full complex spectrum, size = nx*ny */
    d->scale   = 1.0 / (double)(nx * ny);
    d->strength = demag_strength;

    /* ── Step 1: compute real-space demag tensor on CPU (calt) ─────────── */
    printf("[Demag] Computing calt tensor (9 components, 81-point averaging)...\n");

    double *taa = (double*)calloc(d->ncell, sizeof(double));
    double *tab = (double*)calloc(d->ncell, sizeof(double));
    double *tac = (double*)calloc(d->ncell, sizeof(double));
    double *tba = (double*)calloc(d->ncell, sizeof(double));
    double *tbb = (double*)calloc(d->ncell, sizeof(double));
    double *tbc = (double*)calloc(d->ncell, sizeof(double));
    double *tca = (double*)calloc(d->ncell, sizeof(double));
    double *tcb = (double*)calloc(d->ncell, sizeof(double));
    double *tcc = (double*)calloc(d->ncell, sizeof(double));

    if (!taa||!tab||!tac||!tba||!tbb||!tbc||!tca||!tcb||!tcc) {
        fprintf(stderr,"[demag] tensor CPU alloc failed\n");
        free(taa);free(tab);free(tac);free(tba);free(tbb);
        free(tbc);free(tca);free(tcb);free(tcc);free(d);
        return NULL;
    }

    calt(thick, nx, ny, taa,tab,tac, tba,tbb,tbc, tca,tcb,tcc);
    printf("[Demag] calt done.\n");

    /* ── Step 2: pack into complex host arrays (imag = 0, as in pseudocode) */
    /*  Professor's code:
     *    hfaa[idx].x = faa[idx];
     *    hfaa[idx].y = 0.0;
     *  We do the same for all 9 components.
     */
    cufftDoubleComplex *htensor[9];
    double *tptrs[9] = {taa,tab,tac,tba,tbb,tbc,tca,tcb,tcc};
    for (int c = 0; c < 9; c++) {
        htensor[c] = (cufftDoubleComplex*)malloc(
                         (size_t)d->ncell * sizeof(cufftDoubleComplex));
        if (!htensor[c]) {
            fprintf(stderr,"[demag] htensor[%d] alloc failed\n", c);
            /* cleanup */
            for (int cc=0; cc<c; cc++) free(htensor[cc]);
            free(taa);free(tab);free(tac);free(tba);free(tbb);
            free(tbc);free(tca);free(tcb);free(tcc);free(d);
            return NULL;
        }
        for (int idx = 0; idx < d->ncell; idx++) {
            htensor[c][idx].x = tptrs[c][idx];
            htensor[c][idx].y = 0.0;
        }
    }
    free(taa);free(tab);free(tac);free(tba);free(tbb);
    free(tbc);free(tca);free(tcb);free(tcc);

    /* ── Step 3: cuFFT plan  (Z2Z, matches professor's CUFFT_Z2Z plan) ─── */
    /*
     * Professor:  cufftPlan2d(&plan, nn, nn, CUFFT_Z2Z)
     * We use:     cufftPlan2d(&plan, ny, nx, CUFFT_Z2Z)
     * cuFFT 2D convention: first arg = slower (row) dim = ny.
     */
    {
        cufftResult r = cufftPlan2d(&d->plan, ny, nx, CUFFT_Z2Z);
        if (r != CUFFT_SUCCESS) {
            fprintf(stderr,"[demag] cufftPlan2d failed: %d\n",(int)r);
            for (int c=0;c<9;c++) free(htensor[c]);
            free(d); return NULL;
        }
    }

    /* ── Step 4: allocate device arrays ───────────────────────────────── */
    size_t csz = (size_t)d->ncell * sizeof(cufftDoubleComplex);

    for (int c = 0; c < 9; c++) {
        if (cudaMalloc((void**)&d->d_Nhat[c], csz) != cudaSuccess) {
            fprintf(stderr,"[demag] cudaMalloc d_Nhat[%d] failed\n",c);
            for(int cc=0;cc<c;cc++) cudaFree(d->d_Nhat[cc]);
            for(int cc=0;cc<9;cc++) free(htensor[cc]);
            cufftDestroy(d->plan); free(d); return NULL;
        }
    }
    for (int c = 0; c < 3; c++) {
        cudaMalloc((void**)&d->d_Mhat[c], csz);
        cudaMalloc((void**)&d->d_Hhat[c], csz);
    }
    cudaMalloc((void**)&d->d_buf, csz);

    /* ── Step 5: H2D copy tensor, then FFT each component ─────────────── */
    for (int c = 0; c < 9; c++) {
        /* copy real-space tensor component to device */
        cudaMemcpy(d->d_Nhat[c], htensor[c], csz, cudaMemcpyHostToDevice);
        free(htensor[c]);

        /* FFT in-place: d_Nhat[c] = FFT(d_Nhat[c])
         * Professor:  cufftExecZ2Z(plan, d_in, d_out, CUFFT_FORWARD)
         * We do in-place so d_Nhat[c] stays as the output buffer.
         */
        cufftResult r = cufftExecZ2Z(d->plan,
                                     d->d_Nhat[c],
                                     d->d_Nhat[c],
                                     CUFFT_FORWARD);
        if (r != CUFFT_SUCCESS)
            fprintf(stderr,"[demag] tensor FFT[%d] failed: %d\n",c,(int)r);
    }
    cudaDeviceSynchronize();

    /* ── Report ─────────────────────────────────────────────────────────── */
    size_t mem_N   = 9  * csz;
    size_t mem_MH  = 6  * csz;
    size_t mem_buf = csz;
    printf("[Demag] Device memory: N̂=%.1f MB  M̂Ĥ=%.1f MB  buf=%.1f MB\n",
           mem_N/1e6, mem_MH/1e6, mem_buf/1e6);
    printf("[Demag] Ready. strength=%.4f\n", demag_strength);

    return d;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Demag_Apply
 *
 * Called every RHS evaluation.
 * Adds h_dmag = N * M (convolution via FFT) to h_out.
 * ═══════════════════════════════════════════════════════════════════════════ */
void Demag_Apply(DemagData *d, const double *y_dev, double *h_out)
{
    if (!d) return;

    const int ncell = d->ncell;
    const int block = 256;
    const int grid  = (ncell + block - 1) / block;

    /* ── Step 1: FFT each magnetization component ───────────────────────
     * gather m_α from SoA → complex buffer (imag=0) → forward FFT → M̂_α
     * Matches professor's:
     *   h_in[idx].x = mx[idx]; h_in[idx].y = 0;
     *   cufftExecZ2Z(plan, h_in_dev, M_hat, CUFFT_FORWARD)
     */
    for (int comp = 0; comp < 3; comp++) {
        gather_to_complex_kernel<<<grid, block>>>(
            y_dev, d->d_Mhat[comp], comp, ncell);

        cufftExecZ2Z(d->plan,
                     d->d_Mhat[comp],
                     d->d_Mhat[comp],
                     CUFFT_FORWARD);
    }

    /* ── Step 2: pointwise tensor multiply in k-space ───────────────────
     * Ĥ_α(k) = Σ_β  N̂_αβ(k) · M̂_β(k)
     */
    {
        const int g = (d->nk + block - 1) / block;
        pointwise_multiply_kernel<<<g, block>>>(
            d->d_Mhat[0], d->d_Mhat[1], d->d_Mhat[2],
            d->d_Nhat[0], d->d_Nhat[1], d->d_Nhat[2],   /* aa,ab,ac */
            d->d_Nhat[3], d->d_Nhat[4], d->d_Nhat[5],   /* ba,bb,bc */
            d->d_Nhat[6], d->d_Nhat[7], d->d_Nhat[8],   /* ca,cb,cc */
            d->d_Hhat[0], d->d_Hhat[1], d->d_Hhat[2],
            d->nk);
    }

    /* ── Step 3: IFFT each field component → scatter-add to h_out ───────
     * Professor:
     *   cufftExecZ2Z(plan, Hhat, h_buf, CUFFT_INVERSE)
     *   result /= (nn*nn)     ← normalization
     * We do it in-place on d_Hhat, then scatter the real part.
     */
    const double s = d->scale * d->strength;

    for (int comp = 0; comp < 3; comp++) {
        cufftExecZ2Z(d->plan,
                     d->d_Hhat[comp],
                     d->d_Hhat[comp],
                     CUFFT_INVERSE);

        scatter_add_real_kernel<<<grid, block>>>(
            d->d_Hhat[comp], h_out, comp, ncell, s);
    }
    /* No explicit sync — CVODE's next CUDA call on stream 0 provides order */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Demag_Destroy
 * ═══════════════════════════════════════════════════════════════════════════ */
void Demag_Destroy(DemagData *d)
{
    if (!d) return;
    cufftDestroy(d->plan);
    for (int c = 0; c < 9; c++) if (d->d_Nhat[c]) cudaFree(d->d_Nhat[c]);
    for (int c = 0; c < 3; c++) {
        if (d->d_Mhat[c]) cudaFree(d->d_Mhat[c]);
        if (d->d_Hhat[c]) cudaFree(d->d_Hhat[c]);
    }
}