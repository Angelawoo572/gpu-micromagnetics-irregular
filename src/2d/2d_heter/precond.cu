/*
 * precond.cu  (v2)
 *
 * Fix: separate J storage from P^{-1}.
 *
 * Root cause of v1 failure
 * Profiling showed psetup_kernel launched only 266 times vs
 * psolve_kernel 20,207 times (nni~3753).  This means CVODE was
 * sending jok=SUNTRUE ~93% of the time, and v1 returned immediately,
 * reusing stale P^{-1} computed with an old gamma.
 *
 * CVODE's jok protocol:
 *   jok=SUNFALSE: y has changed significantly; rebuild J from scratch.
 *   jok=SUNTRUE : J structure is still valid (y hasn't moved much),
 *                 but gamma CAN change every Newton step.
 *                 => Must recompute P = I - gamma*J even when jok=TRUE.
 *
 * v2 solution: two-kernel approach
 * Kernel 1  build_J_kernel   : reads y, computes J(y), stores to d_J.
 *                               Called only when jok=SUNFALSE.
 *
 * Kernel 2  build_Pinv_kernel : reads d_J and gamma, computes
 *                               P = I - gamma*J, inverts via Cramer,
 *                               stores P^{-1} to d_Pinv.
 *                               Called on EVERY psetup (jok=TRUE or FALSE).
 *
 * This separates the expensive stencil read (build_J_kernel, ~85µs,
 * same as RHS) from the cheap matrix arithmetic (build_Pinv_kernel,
 * no global memory reads of y, just 9 doubles/cell from d_J).
 *
 * Expected outcome
 * psetup now always produces P^{-1} consistent with the current gamma.
 * GMRES should converge in 2-3 iterations instead of 5, reducing:
 *   - psolve_kernel calls by ~50%
 *   - linearSumKernel calls by ~50%
 *   - dotProdKernel calls by ~50%
 *   - total runtime by ~20-30%
 */

#include "precond.h"

#include <cvode/cvode.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Physical constants — must match 2d_p.cu */
__constant__ sunrealtype pc_msk[3]  = {0.0, 0.0, 1.0};
__constant__ sunrealtype pc_nsk[3]  = {1.0, 0.0, 0.0};
__constant__ sunrealtype pc_chk     = 1.0;
__constant__ sunrealtype pc_che     = 4.0;
__constant__ sunrealtype pc_alpha   = 0.2;
__constant__ sunrealtype pc_chg     = 1.0;
__constant__ sunrealtype pc_cha     = 0.0;
__constant__ sunrealtype pc_chb     = 0.3;

/* Index / wrap helpers */
__device__ static inline int pidx_mx(int c, int nc) { return c; }
__device__ static inline int pidx_my(int c, int nc) { return nc + c; }
__device__ static inline int pidx_mz(int c, int nc) { return 2*nc + c; }

__device__ static inline int pwrap_x(int x, int ng) {
    return (x < 0) ? (x+ng) : ((x >= ng) ? (x-ng) : x);
}
__device__ static inline int pwrap_y(int y, int ny) {
    return (y < 0) ? (y+ny) : ((y >= ny) ? (y-ny) : y);
}

/* Kernel 1: build_J_kernel
 *
 * Computes the analytic 3×3 local Jacobian J[i] for each cell i
 * and stores it in d_J[9*cell .. 9*cell+8] (row-major).
 *
 * Called only when jok=SUNFALSE (y has changed significantly).
 * This is the expensive kernel: it reads y and its 4 neighbors.
 *
 * J[row][col] = ∂f_row/∂m_col  (self-coupling only, neighbors frozen)
 * See precond.h / v1 comments for full derivation. */
__global__ static void build_J_kernel(
    const sunrealtype* __restrict__ y,
    sunrealtype*       __restrict__ d_J,   /* 9 * ncell */
    int ng, int ny, int ncell)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= ng || gy >= ny) return;

    const int cell = gy * ng + gx;

    const sunrealtype m1 = y[pidx_mx(cell, ncell)];
    const sunrealtype m2 = y[pidx_my(cell, ncell)];
    const sunrealtype m3 = y[pidx_mz(cell, ncell)];

    /* neighbor cells */
    const int xl  = pwrap_x(gx-1, ng),  xr  = pwrap_x(gx+1, ng);
    const int yu  = pwrap_y(gy-1, ny),  ydn = pwrap_y(gy+1, ny);
    const int lc  = gy*ng + xl,          rc  = gy*ng + xr;
    const int uc  = yu*ng + gx,          dc  = ydn*ng + gx;

    /* h_eff (identical to RHS kernel) */
    const sunrealtype h1 =
        pc_che * (y[pidx_mx(lc,ncell)] + y[pidx_mx(rc,ncell)] +
                  y[pidx_mx(uc,ncell)] + y[pidx_mx(dc,ncell)]) +
        pc_msk[0] * (pc_chk*m3 + pc_cha) +
        pc_chb * pc_nsk[0] * (y[pidx_mx(lc,ncell)] + y[pidx_mx(rc,ncell)]);

    const sunrealtype h2 =
        pc_che * (y[pidx_my(lc,ncell)] + y[pidx_my(rc,ncell)] +
                  y[pidx_my(uc,ncell)] + y[pidx_my(dc,ncell)]) +
        pc_msk[1] * (pc_chk*m3 + pc_cha) +
        pc_chb * pc_nsk[1] * (y[pidx_my(lc,ncell)] + y[pidx_my(rc,ncell)]);

    const sunrealtype h3 =
        pc_che * (y[pidx_mz(lc,ncell)] + y[pidx_mz(rc,ncell)] +
                  y[pidx_mz(uc,ncell)] + y[pidx_mz(dc,ncell)]) +
        pc_msk[2] * (pc_chk*m3 + pc_cha) +
        pc_chb * pc_nsk[2] * (y[pidx_mz(lc,ncell)] + y[pidx_mz(rc,ncell)]);

    const sunrealtype mh = m1*h1 + m2*h2 + m3*h3;

    /* ∂mh/∂mi */
    const sunrealtype d1 = h1;
    const sunrealtype d2 = h2;
    const sunrealtype d3 = h3 + pc_chk*m3;

    /* Store J row-major in d_J[9*cell .. 9*cell+8] */
    const int b = cell * 9;
    d_J[b+0] = pc_alpha * (-d1*m1 - mh);
    d_J[b+1] = -pc_chg*h3 - pc_alpha*d2*m1;
    d_J[b+2] =  pc_chg*(h2 - m2) - pc_alpha*d3*m1;

    d_J[b+3] =  pc_chg*h3 - pc_alpha*d1*m2;
    d_J[b+4] = pc_alpha * (-d2*m2 - mh);
    d_J[b+5] =  pc_chg*(m1 - h1) - pc_alpha*d3*m2;

    d_J[b+6] = -pc_chg*h2 - pc_alpha*d1*m3;
    d_J[b+7] =  pc_chg*h1 - pc_alpha*d2*m3;
    d_J[b+8] = pc_alpha * (pc_msk[2]*pc_chk - d3*m3 - mh);
}

/* Kernel 2: build_Pinv_kernel
 *
 * Reads d_J (the stored Jacobian) and the current gamma.
 * Computes P = I - gamma * J per cell and stores P^{-1}
 * via Cramer's rule into d_Pinv.
 *
 * Called on EVERY psetup (jok=TRUE or jok=FALSE), because
 * gamma changes every Newton step.
 *
 * This kernel is cheap: only reads 9 doubles/cell from d_J,
 * no y stencil reads. Purely arithmetic. */
__global__ static void build_Pinv_kernel(
    const sunrealtype* __restrict__ d_J,
    sunrealtype                     gamma,
    sunrealtype*       __restrict__ d_Pinv,
    int ncell)
{
    const int cell = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (cell >= ncell) return;

    const int b = cell * 9;

    /* P = I - gamma * J */
    const sunrealtype P00 = 1.0 - gamma*d_J[b+0];
    const sunrealtype P01 =      -gamma*d_J[b+1];
    const sunrealtype P02 =      -gamma*d_J[b+2];
    const sunrealtype P10 =      -gamma*d_J[b+3];
    const sunrealtype P11 = 1.0 - gamma*d_J[b+4];
    const sunrealtype P12 =      -gamma*d_J[b+5];
    const sunrealtype P20 =      -gamma*d_J[b+6];
    const sunrealtype P21 =      -gamma*d_J[b+7];
    const sunrealtype P22 = 1.0 - gamma*d_J[b+8];

    /* det(P) */
    const sunrealtype det = P00*(P11*P22 - P12*P21)
                          - P01*(P10*P22 - P12*P20)
                          + P02*(P10*P21 - P11*P20);

    const sunrealtype inv_det = (det != 0.0) ? (1.0 / det) : 1.0;

    /* P^{-1} = (1/det) * adj(P) — stored row-major */
    d_Pinv[b+0] =  inv_det * (P11*P22 - P12*P21);
    d_Pinv[b+1] = -inv_det * (P01*P22 - P02*P21);
    d_Pinv[b+2] =  inv_det * (P01*P12 - P02*P11);
    d_Pinv[b+3] = -inv_det * (P10*P22 - P12*P20);
    d_Pinv[b+4] =  inv_det * (P00*P22 - P02*P20);
    d_Pinv[b+5] = -inv_det * (P00*P12 - P02*P10);
    d_Pinv[b+6] =  inv_det * (P10*P21 - P11*P20);
    d_Pinv[b+7] = -inv_det * (P00*P21 - P01*P20);
    d_Pinv[b+8] =  inv_det * (P00*P11 - P01*P10);
}

/* Kernel 3: psolve_kernel — unchanged from v1 */
__global__ static void psolve_kernel(
    const sunrealtype* __restrict__ r,
    sunrealtype*       __restrict__ z,
    const sunrealtype* __restrict__ Pinv,
    int ncell)
{
    const int cell = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (cell >= ncell) return;

    const int i0 = pidx_mx(cell, ncell);
    const int i1 = pidx_my(cell, ncell);
    const int i2 = pidx_mz(cell, ncell);

    const sunrealtype r0 = r[i0];
    const sunrealtype r1 = r[i1];
    const sunrealtype r2 = r[i2];

    const int b = cell * 9;
    z[i0] = Pinv[b+0]*r0 + Pinv[b+1]*r1 + Pinv[b+2]*r2;
    z[i1] = Pinv[b+3]*r0 + Pinv[b+4]*r1 + Pinv[b+5]*r2;
    z[i2] = Pinv[b+6]*r0 + Pinv[b+7]*r1 + Pinv[b+8]*r2;
}

/* Public API */
PrecondData* Precond_Create(int ng, int ny, int ncell)
{
    PrecondData *pd = (PrecondData*)malloc(sizeof(PrecondData));
    if (!pd) { fprintf(stderr, "precond: malloc failed\n"); return NULL; }

    pd->ng         = ng;
    pd->ny         = ny;
    pd->ncell      = ncell;
    pd->last_gamma = 0.0;
    pd->d_J        = NULL;
    pd->d_Pinv     = NULL;

    const size_t sz = (size_t)ncell * 9 * sizeof(sunrealtype);

    if (cudaMalloc((void**)&pd->d_J,    sz) != cudaSuccess ||
        cudaMalloc((void**)&pd->d_Pinv, sz) != cudaSuccess) {
        fprintf(stderr, "precond: cudaMalloc failed\n");
        Precond_Destroy(pd);
        return NULL;
    }

    printf("[Precond v2] 3x3 block-diagonal Jacobi preconditioner allocated.\n");
    printf("[Precond v2] Storage: %zu MB (J) + %zu MB (Pinv) = %zu MB total\n",
           sz/(1024*1024), sz/(1024*1024), 2*sz/(1024*1024));
    printf("[Precond v2] Fix: gamma updated on EVERY psetup call (jok=TRUE included).\n");

    return pd;
}

void Precond_Destroy(PrecondData *pd)
{
    if (!pd) return;
    if (pd->d_J)    cudaFree(pd->d_J);
    if (pd->d_Pinv) cudaFree(pd->d_Pinv);
    free(pd);
}

/* PrecondSetup — CVODE psetup callback
 *
 * Two-phase approach:
 *   Phase 1 (jok=SUNFALSE): rebuild J from y (expensive stencil read)
 *   Phase 2 (always):       rebuild P^{-1} from J and current gamma
 *
 * This ensures P^{-1} always matches the current gamma, which is
 * the key fix over v1. */
int PrecondSetup(sunrealtype t,
                 N_Vector y, N_Vector fy,
                 sunbooleantype jok, sunbooleantype *jcurPtr,
                 sunrealtype gamma, void *user_data)
{
    (void)t; (void)fy;

    PrecondData *pd = *(PrecondData**)user_data;

    /* ---- Phase 1: rebuild J if y has changed significantly ---- */
    if (!jok) {
        /* Jacobian structure outdated — recompute from current y */
        const sunrealtype *yd = N_VGetDeviceArrayPointer_Cuda(y);

        const dim3 block2d(16, 8);
        const dim3 grid2d((pd->ng + 15) / 16, (pd->ny + 7) / 8);

        build_J_kernel<<<grid2d, block2d>>>(yd, pd->d_J,
                                            pd->ng, pd->ny, pd->ncell);

        if (cudaPeekAtLastError() != cudaSuccess) {
            fprintf(stderr, "precond: build_J_kernel failed\n");
            return -1;
        }
        *jcurPtr = SUNTRUE;
    } else {
        /* J structure still valid — tell CVODE we didn't recompute J */
        *jcurPtr = SUNFALSE;
    }

    /* ---- Phase 2: ALWAYS rebuild P^{-1} = (I - gamma*J)^{-1} ----
     *
     * This is the critical fix: even when jok=SUNTRUE (J unchanged),
     * gamma changes every Newton step, so P^{-1} must be updated.
     * build_Pinv_kernel only reads d_J (9 doubles/cell, no y access)
     * and is much cheaper than build_J_kernel.
     */
    const int block1d = 256;
    const int grid1d  = (pd->ncell + block1d - 1) / block1d;

    build_Pinv_kernel<<<grid1d, block1d>>>(pd->d_J, gamma, pd->d_Pinv,
                                           pd->ncell);

    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "precond: build_Pinv_kernel failed\n");
        return -1;
    }

    pd->last_gamma = gamma;
    return 0;
}

/* PrecondSolve */
int PrecondSolve(sunrealtype t,
                 N_Vector y, N_Vector fy,
                 N_Vector r, N_Vector z,
                 sunrealtype gamma, sunrealtype delta,
                 int lr, void *user_data)
{
    (void)t; (void)y; (void)fy; (void)gamma; (void)delta; (void)lr;

    PrecondData *pd = *(PrecondData**)user_data;

    const sunrealtype *rd = N_VGetDeviceArrayPointer_Cuda(r);
    sunrealtype       *zd = N_VGetDeviceArrayPointer_Cuda(z);

    const int block = 256;
    const int grid  = (pd->ncell + block - 1) / block;

    psolve_kernel<<<grid, block>>>(rd, zd, pd->d_Pinv, pd->ncell);

    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "precond: psolve_kernel failed\n");
        return -1;
    }

    return 0;
}
