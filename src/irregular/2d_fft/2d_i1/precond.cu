/*
 * precond.cu  (v2, x-axis anisotropy)
 *
 * 3×3 Block-Diagonal Jacobi Preconditioner — x-axis easy axis version.
 *
 * ── What changed from z-axis version ────────────────────────────────────────
 *
 * With c_msk={1,0,0} (x-axis anisotropy):
 *
 *   Effective field:
 *     h1 = (c_che+c_chb)*(yL_x+yR_x) + c_che*(yU_x+yD_x) + c_chk*m1 + c_cha
 *     h2 = c_che*(yL_y+yR_y+yU_y+yD_y)
 *     h3 = c_che*(yL_z+yR_z+yU_z+yD_z)
 *
 * Self-coupling derivatives (holding neighbors frozen):
 *     ∂h1/∂m1 = c_chk          ← anisotropy self-coupling now on component 1
 *     ∂h2/∂m1 = 0
 *     ∂h3/∂m1 = 0
 *     (all other ∂hi/∂mj self = 0)
 *
 * Therefore:
 *     d1 = h1 + c_chk    (∂(m·h)/∂m1 = h1 + m1*∂h1/∂m1 = h1 + c_chk*m1... wait)
 *
 * Full self-coupling Jacobian:
 *     ∂(m·h)/∂m1 = h1 + m1*(c_chk)      = d1
 *     ∂(m·h)/∂m2 = h2                    = d2
 *     ∂(m·h)/∂m3 = h3                    = d3   (no anisotropy term here!)
 *
 * So:
 *     d1 = h1 + c_chk*m1    [x-axis: anisotropy contributes to d1]
 *     d2 = h2
 *     d3 = h3               [z-axis version had: d3 = h3 + c_chk*m3]
 *
 * Jacobian blocks (self-coupling only, neighbors frozen):
 *
 *   J[0][0] = c_alpha*(-d1*m1 - mh) + m1*c_alpha*(-c_chk*m1)... 
 *
 * Wait — let's be more careful. The full self-coupling for J[row][col]:
 *
 *   f1 = c_chg*(m3*h2 - m2*h3) + c_alpha*(h1 - mh*m1)
 *   f2 = c_chg*(m1*h3 - m3*h1) + c_alpha*(h2 - mh*m2)
 *   f3 = c_chg*(m2*h1 - m1*h2) + c_alpha*(h3 - mh*m3)
 *
 * ∂f1/∂m1 = c_alpha*( ∂h1/∂m1 - (∂mh/∂m1)*m1 - mh )
 *          = c_alpha*( c_chk - d1*m1 - mh )
 *          [because ∂h1/∂m1 = c_chk, ∂mh/∂m1 = d1 = h1 + c_chk*m1]
 *   But in the block-diagonal approximation we group it as:
 *          = c_alpha*(-d1*m1 - mh)
 *   ... and separately track the c_chk contribution through d1.
 *
 * Actually the cleanest way: keep the SAME formula as the z-axis version
 * but swap which component carries c_chk:
 *
 *   z-axis: d3 = h3 + c_chk*m3,  d1=h1, d2=h2
 *   x-axis: d1 = h1 + c_chk*m1,  d2=h2, d3=h3   ← only this line changes
 *
 * The J[row][col] formulas below are identical in structure; only d1/d2/d3
 * carry different values.
 *
 * Full block:
 *   J[0][0] = c_alpha*(-d1*m1 - mh)
 *   J[0][1] = -c_chg*h3 - c_alpha*d2*m1
 *   J[0][2] =  c_chg*h2 - c_alpha*d3*m1       ← no m2 term (x-axis)
 *
 *   J[1][0] =  c_chg*h3 - c_alpha*d1*m2
 *   J[1][1] = c_alpha*(-d2*m2 - mh)
 *   J[1][2] = -c_chg*h1 - c_alpha*d3*m2       ← no m1 term (x-axis)
 *
 *   J[2][0] = -c_chg*h2 - c_alpha*d1*m3
 *   J[2][1] =  c_chg*h1 - c_alpha*d2*m3
 *   J[2][2] = c_alpha*(-d3*m3 - mh)           ← no c_chk self term here
 *
 * Compare z-axis version:
 *   J[0][2] =  c_chg*(h2 - m2) - c_alpha*d3*m1   ← had extra -m2 from ∂h3/∂m3
 *   J[1][2] =  c_chg*(m1 - h1) - c_alpha*d3*m2   ← had extra +m1
 *   J[2][2] = c_alpha*(c_chk - d3*m3 - mh)       ← had c_chk here
 *
 * In x-axis version those cross terms move to J[*][0]:
 *   J[0][0]: via d1 = h1 + c_chk*m1, no extra explicit term needed
 *   J[1][0] =  c_chg*h3 - c_alpha*d1*m2          (was c_chg*h3 - c_alpha*d1*m2, same)
 *   J[2][0] = -c_chg*h2 - c_alpha*d1*m3          (was -c_chg*h2 - c_alpha*d1*m3, same)
 *
 * The cross-gyro terms from ∂hi/∂mj are:
 *   x-axis anisotropy: ∂h1/∂m1=c_chk only, no off-diagonal ∂hi/∂mj from anisotropy.
 *   So the gyro correction that was "+ c_chg*(h2-m2)" in z-axis (from ∂h3/∂m3 in
 *   the gyro term m2*h3) does NOT appear in x-axis version.
 *
 * Summary of changes in build_J_kernel:
 *   d1 = h1 + c_chk * m1    (was: d1 = h1)
 *   d3 = h3                  (was: d3 = h3 + c_chk * m3)
 *   J[0][2] = c_chg*h2 - c_alpha*d3*m1          (was: c_chg*(h2-m2) - c_alpha*d3*m1)
 *   J[1][2] = -c_chg*h1 - c_alpha*d3*m2         (was: c_chg*(m1-h1) - c_alpha*d3*m2)
 *   J[2][2] = c_alpha*(-d3*m3 - mh)             (was: c_alpha*(c_chk - d3*m3 - mh))
 */

#include "precond.h"

#include <cvode/cvode.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Physical constants — must match 2d_fft.cu (x-axis anisotropy) */
__constant__ sunrealtype pc_msk[3]  = {1.0, 0.0, 0.0};   /* x-axis easy axis */
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

/*
 * Kernel 1: build_J_kernel  (x-axis anisotropy version)
 *
 * Computes analytic 3×3 local Jacobian J[i] per cell and stores to
 * d_J[9*cell .. 9*cell+8] (row-major).
 *
 * Called only when jok=SUNFALSE.
 *
 * Key difference from z-axis version:
 *   d1 = h1 + c_chk*m1    (anisotropy self-coupling on component 1)
 *   d3 = h3               (no anisotropy self-coupling on component 3)
 *
 *   J[0][2], J[1][2], J[2][2] lose the gyro correction terms that came
 *   from ∂h3/∂m3 in the z-axis version.
 */
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

    /*
     * h_eff at current y (identical to RHS kernel with c_msk={1,0,0}):
     *   h1: exchange + DMI/chb on x-neighbors + x-anisotropy (c_chk*m1)
     *   h2: exchange only
     *   h3: exchange only
     */
    const sunrealtype h1 =
        (pc_che + pc_chb) * (y[pidx_mx(lc,ncell)] + y[pidx_mx(rc,ncell)]) +
        pc_che * (y[pidx_mx(uc,ncell)] + y[pidx_mx(dc,ncell)]) +
        pc_chk * m1 + pc_cha;

    const sunrealtype h2 =
        pc_che * (y[pidx_my(lc,ncell)] + y[pidx_my(rc,ncell)] +
                  y[pidx_my(uc,ncell)] + y[pidx_my(dc,ncell)]);

    const sunrealtype h3 =
        pc_che * (y[pidx_mz(lc,ncell)] + y[pidx_mz(rc,ncell)] +
                  y[pidx_mz(uc,ncell)] + y[pidx_mz(dc,ncell)]);

    const sunrealtype mh = m1*h1 + m2*h2 + m3*h3;

    /*
     * ∂(m·h)/∂mi — self-coupling (neighbors frozen):
     *
     *   d1 = h1 + m1 * ∂h1/∂m1 = h1 + m1*c_chk    ← x-axis: c_chk on d1
     *   d2 = h2                                      (no self term)
     *   d3 = h3                                      (no self term, unlike z-axis)
     */
    const sunrealtype d1 = h1 + pc_chk * m1;   /* x-axis anisotropy self */
    const sunrealtype d2 = h2;
    const sunrealtype d3 = h3;                  /* no c_chk here (was d3=h3+chk*m3) */

    /*
     * 3×3 Jacobian block, row-major in d_J[9*cell..9*cell+8]:
     *
     *   Row 0 (∂f1/∂mj):
     *     J[0][0] = c_alpha*(-d1*m1 - mh)
     *     J[0][1] = -c_chg*h3 - c_alpha*d2*m1
     *     J[0][2] =  c_chg*h2 - c_alpha*d3*m1
     *               (no -m2 gyro correction: ∂h3/∂m3=0 in x-axis version)
     *
     *   Row 1 (∂f2/∂mj):
     *     J[1][0] =  c_chg*h3 - c_alpha*d1*m2
     *     J[1][1] = c_alpha*(-d2*m2 - mh)
     *     J[1][2] = -c_chg*h1 - c_alpha*d3*m2
     *               (no +m1 gyro correction)
     *
     *   Row 2 (∂f3/∂mj):
     *     J[2][0] = -c_chg*h2 - c_alpha*d1*m3
     *     J[2][1] =  c_chg*h1 - c_alpha*d2*m3
     *     J[2][2] = c_alpha*(-d3*m3 - mh)
     *               (no +c_chk self term: that's absorbed in d1 now)
     */
    const int b = cell * 9;

    /* Row 0: ∂f1/∂(m1, m2, m3) */
    d_J[b+0] = pc_alpha * (-d1*m1 - mh);
    d_J[b+1] = -pc_chg*h3 - pc_alpha*d2*m1;
    d_J[b+2] =  pc_chg*h2 - pc_alpha*d3*m1;

    /* Row 1: ∂f2/∂(m1, m2, m3) */
    d_J[b+3] =  pc_chg*h3 - pc_alpha*d1*m2;
    d_J[b+4] = pc_alpha * (-d2*m2 - mh);
    d_J[b+5] = -pc_chg*h1 - pc_alpha*d3*m2;

    /* Row 2: ∂f3/∂(m1, m2, m3) */
    d_J[b+6] = -pc_chg*h2 - pc_alpha*d1*m3;
    d_J[b+7] =  pc_chg*h1 - pc_alpha*d2*m3;
    d_J[b+8] = pc_alpha * (-d3*m3 - mh);
}

/*
 * Kernel 2: build_Pinv_kernel
 *
 * Reads d_J and current gamma, computes P = I - gamma*J per cell,
 * inverts via Cramer's rule, stores P^{-1} to d_Pinv.
 *
 * Called on EVERY psetup (jok=TRUE or FALSE) — gamma changes every Newton step.
 * This kernel is cheap: only reads 9 doubles/cell from d_J, no y stencil.
 */
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

/* Kernel 3: psolve_kernel — unchanged */
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

    printf("[Precond v2 x-axis] 3x3 block-diagonal Jacobi preconditioner allocated.\n");
    printf("[Precond v2 x-axis] Storage: %zu MB (J) + %zu MB (Pinv) = %zu MB total\n",
           sz/(1024*1024), sz/(1024*1024), 2*sz/(1024*1024));
    printf("[Precond v2 x-axis] Anisotropy: x-axis (d1=h1+chk*m1, d3=h3).\n");
    printf("[Precond v2 x-axis] gamma updated on EVERY psetup call (jok=TRUE included).\n");

    return pd;
}

void Precond_Destroy(PrecondData *pd)
{
    if (!pd) return;
    if (pd->d_J)    cudaFree(pd->d_J);
    if (pd->d_Pinv) cudaFree(pd->d_Pinv);
    free(pd);
}

/*
 * PrecondSetup — CVODE psetup callback
 *
 * Phase 1 (jok=SUNFALSE): rebuild J from y via build_J_kernel (expensive).
 * Phase 2 (always):       rebuild P^{-1} from J and current gamma (cheap).
 */
int PrecondSetup(sunrealtype t,
                 N_Vector y, N_Vector fy,
                 sunbooleantype jok, sunbooleantype *jcurPtr,
                 sunrealtype gamma, void *user_data)
{
    (void)t; (void)fy;

    PrecondData *pd = *(PrecondData**)user_data;

    /* Phase 1: rebuild J if y has changed significantly */
    if (!jok) {
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
        *jcurPtr = SUNFALSE;
    }

    /* Phase 2: ALWAYS rebuild P^{-1} = (I - gamma*J)^{-1}
     * Cheap: reads only d_J (9 doubles/cell), no y stencil. */
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

/* PrecondSolve — unchanged */
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
