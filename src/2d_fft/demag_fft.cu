/*
 * demag_test.cu
 * Used ONLY to verify demag field output against MATLAB.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_89 demag_test.cu -lcufft -o demag_test
 *
 * Run:
 *   ./demag_test > output.txt
 *
 * Then check in MATLAB.
 *
 * The ONLY change from the original: nn=64 → adjustable via NN below.
 * Everything else — index remapping, printf, algorithm — is untouched.
 */

#include <cstdio>
#include <cmath>
#include <cufft.h>
#include <cuda_runtime.h>

/* ── Change this to test different grid sizes ── */
static const int NN = 64;   /* original tried value */

static inline cufftDoubleComplex cadd(cufftDoubleComplex a, cufftDoubleComplex b)
{
    cufftDoubleComplex r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    return r;
}

static inline cufftDoubleComplex cmul(cufftDoubleComplex a, cufftDoubleComplex b)
{
    cufftDoubleComplex r;
    r.x = a.x * b.x - a.y * b.y;
    r.y = a.x * b.y + a.y * b.x;
    return r;
}

int calt(double thik, int mdx, int mdy,
         double taa[], double tab[], double tac[],
         double tba[], double tbb[], double tbc[],
         double tca[], double tcb[], double tcc[]);

void ctt(double b, double a, double sx, double sy, double dm[]);

int main()
{
    const int nn = NN;
    int demagstatus = 0;
    int idx = 0;
    int i = 0, j = 0;

    double* faa = new double[nn * nn];
    double* fab = new double[nn * nn];
    double* fac = new double[nn * nn];
    double* fba = new double[nn * nn];
    double* fbb = new double[nn * nn];
    double* fbc = new double[nn * nn];
    double* fca = new double[nn * nn];
    double* fcb = new double[nn * nn];
    double* fcc = new double[nn * nn];

    cufftDoubleComplex* hfaa = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hfab = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hfac = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hfba = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hfbb = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hfbc = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hfca = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hfcb = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hfcc = new cufftDoubleComplex[nn * nn];

    cufftDoubleComplex* hofaa = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hofab = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hofac = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hofba = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hofbb = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hofbc = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hofca = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hofcb = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hofcc = new cufftDoubleComplex[nn * nn];

    cufftDoubleComplex* hrha = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hrhb = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hrhc = new cufftDoubleComplex[nn * nn];

    cufftDoubleComplex* hrhx = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hrhy = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hrhz = new cufftDoubleComplex[nn * nn];

    int nn8 = nn / 8;
    int nn2 = nn / 2;

    double thick = 100.0;

    // Calculate FFT of Demagnetization Matrix
    demagstatus = calt(thick, nn, nn,
                       faa, fab, fac,
                       fba, fbb, fbc,
                       fca, fcb, fcc);

    for (int j = 0; j < nn; j++) {
        for (int i = 0; i < nn; i++) {
            idx = j * nn + i;
            hfaa[idx].x = faa[idx];
            hfab[idx].x = fab[idx];
            hfac[idx].x = fac[idx];
            hfaa[idx].y = 0.0;
            hfab[idx].y = 0.0;
            hfac[idx].y = 0.0;

            hfba[idx].x = fba[idx];
            hfbb[idx].x = fbb[idx];
            hfbc[idx].x = fbc[idx];
            hfba[idx].y = 0.0;
            hfbb[idx].y = 0.0;
            hfbc[idx].y = 0.0;

            hfca[idx].x = fca[idx];
            hfcb[idx].x = fcb[idx];
            hfcc[idx].x = fcc[idx];
            hfca[idx].y = 0.0;
            hfcb[idx].y = 0.0;
            hfcc[idx].y = 0.0;
        }
    }

    cufftDoubleComplex* difaa = nullptr;
    cufftDoubleComplex* difab = nullptr;
    cufftDoubleComplex* difac = nullptr;
    cufftDoubleComplex* difba = nullptr;
    cufftDoubleComplex* difbb = nullptr;
    cufftDoubleComplex* difbc = nullptr;
    cufftDoubleComplex* difca = nullptr;
    cufftDoubleComplex* difcb = nullptr;
    cufftDoubleComplex* difcc = nullptr;

    cufftDoubleComplex* dofaa = nullptr;
    cufftDoubleComplex* dofab = nullptr;
    cufftDoubleComplex* dofac = nullptr;
    cufftDoubleComplex* dofba = nullptr;
    cufftDoubleComplex* dofbb = nullptr;
    cufftDoubleComplex* dofbc = nullptr;
    cufftDoubleComplex* dofca = nullptr;
    cufftDoubleComplex* dofcb = nullptr;
    cufftDoubleComplex* dofcc = nullptr;

    cudaError_t cuda_status;

    cuda_status = cudaMalloc((void**)&difaa, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&difab, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&difac, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&difba, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&difbb, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&difbc, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&difca, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&difcb, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&difcc, nn * nn * sizeof(cufftDoubleComplex));

    cuda_status = cudaMalloc((void**)&dofaa, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&dofab, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&dofac, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&dofba, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&dofbb, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&dofbc, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&dofca, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&dofcb, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&dofcc, nn * nn * sizeof(cufftDoubleComplex));

    if (cuda_status != cudaSuccess) {
        std::printf("cudaMalloc d_in failed\n");
        return 1;
    }

    cuda_status = cudaMemcpy(difaa, hfaa, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cuda_status = cudaMemcpy(difab, hfab, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cuda_status = cudaMemcpy(difac, hfac, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cuda_status = cudaMemcpy(difba, hfba, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cuda_status = cudaMemcpy(difbb, hfbb, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cuda_status = cudaMemcpy(difbc, hfbc, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cuda_status = cudaMemcpy(difca, hfca, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cuda_status = cudaMemcpy(difcb, hfcb, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cuda_status = cudaMemcpy(difcc, hfcc, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftResult fft_status;
    fft_status = cufftPlan2d(&plan, nn, nn, CUFFT_Z2Z);

    if (fft_status != CUFFT_SUCCESS) {
        std::printf("cufftPlan2d failed\n");
        return 1;
    }

    fft_status = cufftExecZ2Z(plan, difaa, dofaa, CUFFT_FORWARD);
    fft_status = cufftExecZ2Z(plan, difab, dofab, CUFFT_FORWARD);
    fft_status = cufftExecZ2Z(plan, difac, dofac, CUFFT_FORWARD);
    fft_status = cufftExecZ2Z(plan, difba, dofba, CUFFT_FORWARD);
    fft_status = cufftExecZ2Z(plan, difbb, dofbb, CUFFT_FORWARD);
    fft_status = cufftExecZ2Z(plan, difbc, dofbc, CUFFT_FORWARD);
    fft_status = cufftExecZ2Z(plan, difca, dofca, CUFFT_FORWARD);
    fft_status = cufftExecZ2Z(plan, difcb, dofcb, CUFFT_FORWARD);
    fft_status = cufftExecZ2Z(plan, difcc, dofcc, CUFFT_FORWARD);

    cuda_status = cudaMemcpy(hofaa, dofaa, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cuda_status = cudaMemcpy(hofab, dofab, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cuda_status = cudaMemcpy(hofac, dofac, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cuda_status = cudaMemcpy(hofba, dofba, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cuda_status = cudaMemcpy(hofbb, dofbb, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cuda_status = cudaMemcpy(hofbc, dofbc, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cuda_status = cudaMemcpy(hofca, dofca, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cuda_status = cudaMemcpy(hofcb, dofcb, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cuda_status = cudaMemcpy(hofcc, dofcc, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

    if (fft_status != CUFFT_SUCCESS) {
        std::printf("cufftExecZ2Z failed\n");
        cufftDestroy(plan);
        cudaFree(dofaa); cudaFree(dofab); cudaFree(dofac);
        cudaFree(dofba); cudaFree(dofbb); cudaFree(dofbc);
        cudaFree(dofca); cudaFree(dofcb); cudaFree(dofcc);
        cudaFree(difaa); cudaFree(difab); cudaFree(difac);
        cudaFree(difba); cudaFree(difbb); cudaFree(difbc);
        cudaFree(difca); cudaFree(difcb); cudaFree(difcc);
        return 1;
    }

    double* hma = new double[nn * nn];
    double* hmb = new double[nn * nn];
    double* hmc = new double[nn * nn];

    cufftDoubleComplex* hma_c = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hmb_c = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hmc_c = new cufftDoubleComplex[nn * nn];

    cufftDoubleComplex* homa = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* homb = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* homc = new cufftDoubleComplex[nn * nn];

    cufftDoubleComplex* hkha = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hkhb = new cufftDoubleComplex[nn * nn];
    cufftDoubleComplex* hkhc = new cufftDoubleComplex[nn * nn];

    int i1 = nn / 3;
    int i2 = 2 * nn / 3;
    int j1 = nn / 3;
    int j2 = 2 * nn / 3;

    for (j = 0; j < nn; j++) {
        for (i = 0; i < nn; i++) {
            idx = j * nn + i;
            hma[idx] = 1.0;
            hmb[idx] = 0.0;
            hmc[idx] = 0.0;
            if (i > i1 && i < i2) {
                if (j > j1 && j < j2) {
                    hma[idx] = 0.0;
                }
            }
            hma_c[idx].x = hma[idx];
            hma_c[idx].y = 0.0;
            hmb_c[idx].x = hmb[idx];
            hmb_c[idx].y = 0.0;
            hmc_c[idx].x = hmc[idx];
            hmc_c[idx].y = 0.0;
        }
    }

    // FFT for magnetization ma, mb, mc
    cufftDoubleComplex* dima = nullptr;
    cufftDoubleComplex* dimb = nullptr;
    cufftDoubleComplex* dimc = nullptr;

    cufftDoubleComplex* doma = nullptr;
    cufftDoubleComplex* domb = nullptr;
    cufftDoubleComplex* domc = nullptr;

    cuda_status = cudaMalloc((void**)&dima, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&dimb, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&dimc, nn * nn * sizeof(cufftDoubleComplex));

    cuda_status = cudaMalloc((void**)&doma, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&domb, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&domc, nn * nn * sizeof(cufftDoubleComplex));

    cuda_status = cudaMemcpy(dima, hma_c, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cuda_status = cudaMemcpy(dimb, hmb_c, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cuda_status = cudaMemcpy(dimc, hmc_c, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

    fft_status = cufftExecZ2Z(plan, dima, doma, CUFFT_FORWARD);
    fft_status = cufftExecZ2Z(plan, dimb, domb, CUFFT_FORWARD);
    fft_status = cufftExecZ2Z(plan, dimc, domc, CUFFT_FORWARD);

    cuda_status = cudaMemcpy(homa, doma, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cuda_status = cudaMemcpy(homb, domb, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cuda_status = cudaMemcpy(homc, domc, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

    /* ── pointwise multiply in k-space ── */
    for (j = 0; j < nn; j++) {
        for (i = 0; i < nn; i++) {
            idx = j * nn + i;
            hkha[idx] = cadd(cadd(cmul(hofaa[idx], homa[idx]),
                                  cmul(hofab[idx], homb[idx])),
                                  cmul(hofac[idx], homc[idx]));
            hkhb[idx] = cadd(cadd(cmul(hofba[idx], homa[idx]),
                                  cmul(hofbb[idx], homb[idx])),
                                  cmul(hofbc[idx], homc[idx]));
            hkhc[idx] = cadd(cadd(cmul(hofca[idx], homa[idx]),
                                  cmul(hofcb[idx], homb[idx])),
                                  cmul(hofcc[idx], homc[idx]));
        }
    }

    cufftDoubleComplex* dkha = nullptr;
    cufftDoubleComplex* dkhb = nullptr;
    cufftDoubleComplex* dkhc = nullptr;
    cufftDoubleComplex* drha = nullptr;
    cufftDoubleComplex* drhb = nullptr;
    cufftDoubleComplex* drhc = nullptr;

    cuda_status = cudaMalloc((void**)&dkha, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&dkhb, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&dkhc, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&drha, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&drhb, nn * nn * sizeof(cufftDoubleComplex));
    cuda_status = cudaMalloc((void**)&drhc, nn * nn * sizeof(cufftDoubleComplex));

    cuda_status = cudaMemcpy(dkha, hkha, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cuda_status = cudaMemcpy(dkhb, hkhb, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cuda_status = cudaMemcpy(dkhc, hkhc, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

    // Inverse FFT for Field  dkh -> drh
    fft_status = cufftExecZ2Z(plan, dkha, drha, CUFFT_INVERSE);
    fft_status = cufftExecZ2Z(plan, dkhb, drhb, CUFFT_INVERSE);
    fft_status = cufftExecZ2Z(plan, dkhc, drhc, CUFFT_INVERSE);

    if (cuda_status != cudaSuccess) {
        std::printf("cudaMemcpy D2H failedx\n");
        cufftDestroy(plan);
        return 1;
    }

    cuda_status = cudaMemcpy(hrha, drha, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cuda_status = cudaMemcpy(hrhb, drhb, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cuda_status = cudaMemcpy(hrhc, drhc, nn * nn * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

    /* ── output wit index remapping ── */
    int idxnew = 0;
    int ix = 0;
    int jy = 0;
    std::printf("%d %d %d\n", nn, nn, nn);
    for (int j = 0; j < nn; j++) {
        if (j < nn2) {
            jy = nn2 - j;
        } else {
            jy = nn - j + nn2 - 1;
        }
        for (int i = 0; i < nn; i++) {
            if (i < nn2) {
                ix = nn2 - i;
            } else {
                ix = nn - i + nn2 - 1;
            }
            idx    = j * nn + i;
            idxnew = jy * nn + ix;
            double hx = (hrha[idxnew].x) / (nn * nn);
            double hy = (hrhb[idxnew].x) / (nn * nn);
            double hz = (hrhc[idxnew].x) / (nn * nn);
            std::printf("%f %f %f\n", hx, hy, hz);
        }
    }

    if (cuda_status != cudaSuccess) {
        std::printf("cudaMemcpy D2H failedx\n");
        cufftDestroy(plan);
        return 1;
    }

    cufftDestroy(plan);

    cudaFree(difaa); cudaFree(difab); cudaFree(difac);
    cudaFree(difba); cudaFree(difbb); cudaFree(difbc);
    cudaFree(difca); cudaFree(difcb); cudaFree(difcc);

    cudaFree(dofaa); cudaFree(dofab); cudaFree(dofac);
    cudaFree(dofba); cudaFree(dofbb); cudaFree(dofbc);
    cudaFree(dofca); cudaFree(dofcb); cudaFree(dofcc);

    cudaFree(dima); cudaFree(dimb); cudaFree(dimc);
    cudaFree(doma); cudaFree(domb); cudaFree(domc);

    cudaFree(dkha); cudaFree(dkhb); cudaFree(dkhc);
    cudaFree(drha); cudaFree(drhb); cudaFree(drhc);

    delete[] faa; delete[] fab; delete[] fac;
    delete[] fba; delete[] fbb; delete[] fbc;
    delete[] fca; delete[] fcb; delete[] fcc;

    delete[] hfaa; delete[] hfab; delete[] hfac;
    delete[] hfba; delete[] hfbb; delete[] hfbc;
    delete[] hfca; delete[] hfcb; delete[] hfcc;

    delete[] hofaa; delete[] hofab; delete[] hofac;
    delete[] hofba; delete[] hofbb; delete[] hofbc;
    delete[] hofca; delete[] hofcb; delete[] hofcc;

    delete[] hrha; delete[] hrhb; delete[] hrhc;
    delete[] hrhx; delete[] hrhy; delete[] hrhz;

    delete[] hma; delete[] hmb; delete[] hmc;
    delete[] hma_c; delete[] hmb_c; delete[] hmc_c;
    delete[] homa; delete[] homb; delete[] homc;
    delete[] hkha; delete[] hkhb; delete[] hkhc;

    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * calt — verbatim from pseudocode
 * (dm[] is 1-indexed as in original: dm[1]..dm[9])
 * ═══════════════════════════════════════════════════════════════ */
int calt(double thik, int mdx, int mdy,
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
    double dm[10];   /* 1-indexed, dm[0] unused */

    for (j = 0; j < mdy; j++) {
        for (i = 0; i < mdx; i++) {
            ikn = j * mdx + i;
            taa[ikn] = 0.0; tab[ikn] = 0.0; tac[ikn] = 0.0;
            tba[ikn] = 0.0; tbb[ikn] = 0.0; tbc[ikn] = 0.0;
            tca[ikn] = 0.0; tcb[ikn] = 0.0; tcc[ikn] = 0.0;

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

/* ═══════════════════════════════════════════════════════════════
 * ctt — verbatim from pseudocode
 * (dm[] is 1-indexed as in original: dm[1]..dm[9])
 * ═══════════════════════════════════════════════════════════════ */
void ctt(double b, double a, double sx, double sy, double dm[])
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
