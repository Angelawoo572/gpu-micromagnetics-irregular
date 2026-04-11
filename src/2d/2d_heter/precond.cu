/*
 * precond.cu
 *
 * 3×3 Block-Diagonal Jacobi Preconditioner for the 2D LLG solver.
 *
 * Physics: LLG equation RHS for cell i (SoA layout):
 *
 *   f1 = c_chg*(m3*h2 - m2*h3) + c_alpha*(h1 - mh*m1)
 *   f2 = c_chg*(m1*h3 - m3*h1) + c_alpha*(h2 - mh*m2)
 *   f3 = c_chg*(m2*h1 - m1*h2) + c_alpha*(h3 - mh*m3)
 *
 * where mh = m1*h1 + m2*h2 + m3*h3.
 *
 * Self-dependent terms in h (holding neighbors frozen):
 *   h1_self = c_msk[0] * c_chk * m3 = 0   (c_msk[0]=0)
 *   h2_self = c_msk[1] * c_chk * m3 = 0   (c_msk[1]=0)
 *   h3_self = c_msk[2] * c_chk * m3 = m3  (c_msk[2]=1, c_chk=1)
 *
 * Therefore: ∂h1/∂mj = 0, ∂h2/∂mj = 0, ∂h3/∂m3 = 1, ∂h3/∂m1 = ∂h3/∂m2 = 0
 *
 * Analytic 3×3 block Jacobian J = ∂f/∂m (self-coupling only):
 *
 *   ∂mh/∂m1 = h1            (since ∂h/∂m1 = 0 except h3 has none)
 *   ∂mh/∂m2 = h2
 *   ∂mh/∂m3 = h3 + m3       (since ∂h3/∂m3 = 1)
 *
 *   J[0][0] = c_alpha * (-h1*m1 - mh)
 *   J[0][1] = -c_chg*h3 - c_alpha*h2*m1
 *   J[0][2] = c_chg*(h2 - m2) - c_alpha*(h3+m3)*m1
 *
 *   J[1][0] = c_chg*h3 - c_alpha*h1*m2
 *   J[1][1] = c_alpha * (-h2*m2 - mh)
 *   J[1][2] = c_chg*(m1 - h1) - c_alpha*(h3+m3)*m2
 *
 *   J[2][0] = -c_chg*h2 - c_alpha*h1*m3
 *   J[2][1] =  c_chg*h1 - c_alpha*h2*m3
 *   J[2][2] = c_alpha * (1 - (h3+m3)*m3 - mh)
 *
 * P = I - gamma * J   (3×3 per cell)
 * P^{-1} computed via Cramer's rule (exact for 3×3)
 * Stored: Pinv[9*cell + 0..8] in row-major order
 */

#include "precond.h"

#include <cvode/cvode.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* =========================================================
 * Physical constants — must match 2d_p.cu
 * ========================================================= */
__constant__ sunrealtype pc_msk[3]  = {0.0, 0.0, 1.0};
__constant__ sunrealtype pc_nsk[3]  = {1.0, 0.0, 0.0};
__constant__ sunrealtype pc_chk     = 1.0;
__constant__ sunrealtype pc_che     = 4.0;
__constant__ sunrealtype pc_alpha   = 0.2;
__constant__ sunrealtype pc_chg     = 1.0;
__constant__ sunrealtype pc_cha     = 0.0;
__constant__ sunrealtype pc_chb     = 0.3;

/* =========================================================
 * Index helpers (SoA)
 * ========================================================= */
__device__ static inline int pidx_mx(int c, int nc) { return c; }
__device__ static inline int pidx_my(int c, int nc) { return nc + c; }
__device__ static inline int pidx_mz(int c, int nc) { return 2*nc + c; }

__device__ static inline int pwrap_x(int x, int ng) {
    return (x < 0) ? (x+ng) : ((x >= ng) ? (x-ng) : x);
}
__device__ static inline int pwrap_y(int y, int ny) {
    return (y < 0) ? (y+ny) : ((y >= ny) ? (y-ny) : y);
}

/* =========================================================
 * Kernel: psetup_kernel
 *
 * For each cell:
 *   1. Recompute h_eff (needs neighbor reads, same as RHS)
 *   2. Build local 3×3 Jacobian J from analytic formulas above
 *   3. Compute P = I - gamma * J
 *   4. Invert P via Cramer's rule
 *   5. Store P^{-1} in Pinv[9*cell .. 9*cell+8] (row-major)
 *
 * One thread per cell, 2D launch matching the RHS kernel.
 * ========================================================= */
__global__ static void psetup_kernel(
    const sunrealtype* __restrict__ y,
    sunrealtype                     gamma,
    sunrealtype*       __restrict__ Pinv,   /* 9 * ncell */
    int ng, int ny, int ncell)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= ng || gy >= ny) return;

    const int cell = gy * ng + gx;

    /* fetch self magnetization */
    const sunrealtype m1 = y[pidx_mx(cell, ncell)];
    const sunrealtype m2 = y[pidx_my(cell, ncell)];
    const sunrealtype m3 = y[pidx_mz(cell, ncell)];

    /* neighbor cells */
    const int xl  = pwrap_x(gx-1, ng);
    const int xr  = pwrap_x(gx+1, ng);
    const int yu  = pwrap_y(gy-1, ny);
    const int ydn = pwrap_y(gy+1, ny);

    const int lc = gy*ng + xl,  rc = gy*ng + xr;
    const int uc = yu*ng + gx,  dc = ydn*ng + gx;

    /* compute h_eff (identical to RHS kernel) */
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

    /* ∂mh/∂mi — needed for Jacobian */
    const sunrealtype d_mh_m1 = h1;
    const sunrealtype d_mh_m2 = h2;
    const sunrealtype d_mh_m3 = h3 + pc_chk*m3;   /* h3 + m3 since c_chk=1 */

    /* Analytic 3×3 local Jacobian J[row][col] = ∂f_row / ∂m_col */
    const sunrealtype J00 = pc_alpha * (-d_mh_m1*m1 - mh);
    const sunrealtype J01 = -pc_chg*h3 - pc_alpha*d_mh_m2*m1;
    const sunrealtype J02 =  pc_chg*(h2 - m2) - pc_alpha*d_mh_m3*m1;

    const sunrealtype J10 =  pc_chg*h3 - pc_alpha*d_mh_m1*m2;
    const sunrealtype J11 = pc_alpha * (-d_mh_m2*m2 - mh);
    const sunrealtype J12 =  pc_chg*(m1 - h1) - pc_alpha*d_mh_m3*m2;

    const sunrealtype J20 = -pc_chg*h2 - pc_alpha*d_mh_m1*m3;
    const sunrealtype J21 =  pc_chg*h1 - pc_alpha*d_mh_m2*m3;
    const sunrealtype J22 = pc_alpha * (pc_msk[2]*pc_chk - d_mh_m3*m3 - mh);

    /* P = I - gamma * J */
    const sunrealtype P00 = 1.0 - gamma*J00;
    const sunrealtype P01 =      -gamma*J01;
    const sunrealtype P02 =      -gamma*J02;
    const sunrealtype P10 =      -gamma*J10;
    const sunrealtype P11 = 1.0 - gamma*J11;
    const sunrealtype P12 =      -gamma*J12;
    const sunrealtype P20 =      -gamma*J20;
    const sunrealtype P21 =      -gamma*J21;
    const sunrealtype P22 = 1.0 - gamma*J22;

    /* det(P) via Sarrus / cofactor expansion along row 0 */
    const sunrealtype det = P00*(P11*P22 - P12*P21)
                          - P01*(P10*P22 - P12*P20)
                          + P02*(P10*P21 - P11*P20);

    /* Guard against singular block (shouldn't happen for physical m, h) */
    const sunrealtype inv_det = (det != 0.0) ? (1.0 / det) : 1.0;

    /* P^{-1} = (1/det) * adj(P)
     * adj[i][j] = (-1)^{i+j} * M_ji  (cofactor of P^T)
     * Stored row-major in Pinv[9*cell .. 9*cell+8]              */
    const int b = cell * 9;

    /* Row 0 of P^{-1}: cofactors of column 0 of P */
    Pinv[b+0] =  inv_det * (P11*P22 - P12*P21);
    Pinv[b+1] = -inv_det * (P01*P22 - P02*P21);
    Pinv[b+2] =  inv_det * (P01*P12 - P02*P11);

    /* Row 1 */
    Pinv[b+3] = -inv_det * (P10*P22 - P12*P20);
    Pinv[b+4] =  inv_det * (P00*P22 - P02*P20);
    Pinv[b+5] = -inv_det * (P00*P12 - P02*P10);

    /* Row 2 */
    Pinv[b+6] =  inv_det * (P10*P21 - P11*P20);
    Pinv[b+7] = -inv_det * (P00*P21 - P01*P20);
    Pinv[b+8] =  inv_det * (P00*P11 - P01*P10);
}

/* =========================================================
 * Kernel: psolve_kernel
 *
 * Applies z = P^{-1} r per cell in SoA layout.
 * One thread per cell; reads the 3-vector (r_mx, r_my, r_mz)
 * and multiplies by the stored 3×3 block.
 *
 * This kernel is bandwidth-limited but very simple; it reads
 * 3 doubles from r and 9 doubles from Pinv, writes 3 doubles
 * to z — 15 memory accesses per cell.  For 1.28M cells and
 * 8 bytes/double: ~150 MB per call.  At 1000 GB/s → ~150 µs.
 * This replaces 5+ GMRES iterations of heavy vector ops.
 * ========================================================= */
__global__ static void psolve_kernel(
    const sunrealtype* __restrict__ r,
    sunrealtype*       __restrict__ z,
    const sunrealtype* __restrict__ Pinv,
    int ncell)
{
    const int cell = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (cell >= ncell) return;

    const int i0 = pidx_mx(cell, ncell);  /* = cell          */
    const int i1 = pidx_my(cell, ncell);  /* = ncell + cell  */
    const int i2 = pidx_mz(cell, ncell);  /* = 2*ncell + cell */

    const sunrealtype r0 = r[i0];
    const sunrealtype r1 = r[i1];
    const sunrealtype r2 = r[i2];

    const int b = cell * 9;

    z[i0] = Pinv[b+0]*r0 + Pinv[b+1]*r1 + Pinv[b+2]*r2;
    z[i1] = Pinv[b+3]*r0 + Pinv[b+4]*r1 + Pinv[b+5]*r2;
    z[i2] = Pinv[b+6]*r0 + Pinv[b+7]*r1 + Pinv[b+8]*r2;
}

/* =========================================================
 * Public API
 * ========================================================= */

PrecondData* Precond_Create(int ng, int ny, int ncell)
{
    PrecondData *pd = (PrecondData*)malloc(sizeof(PrecondData));
    if (!pd) { fprintf(stderr, "precond: malloc failed\n"); return NULL; }

    pd->ng    = ng;
    pd->ny    = ny;
    pd->ncell = ncell;

    cudaError_t e = cudaMalloc((void**)&pd->d_Pinv,
                               (size_t)ncell * 9 * sizeof(sunrealtype));
    if (e != cudaSuccess) {
        fprintf(stderr, "precond: cudaMalloc Pinv failed: %s\n",
                cudaGetErrorString(e));
        free(pd);
        return NULL;
    }

    printf("[Precond] 3x3 block-diagonal Jacobi preconditioner allocated.\n");
    printf("[Precond] Storage: %zu MB\n",
           (size_t)ncell * 9 * sizeof(sunrealtype) / (1024*1024));

    return pd;
}

void Precond_Destroy(PrecondData *pd)
{
    if (!pd) return;
    if (pd->d_Pinv) cudaFree(pd->d_Pinv);
    free(pd);
}

/* =========================================================
 * CVODE psetup callback
 *
 * Signature required by CVODE:
 *   int psetup(t, y, fy, jok, jcurPtr, gamma, user_data)
 *
 * user_data points to UserDataFull (defined in 2d_p.cu),
 * whose first member is PrecondData *pd.
 * ========================================================= */
int PrecondSetup(sunrealtype t,
                 N_Vector y, N_Vector fy,
                 sunbooleantype jok, sunbooleantype *jcurPtr,
                 sunrealtype gamma, void *user_data)
{
    (void)t; (void)fy;

    /* UserDataFull has PrecondData* as first field */
    PrecondData *pd = *(PrecondData**)user_data;

    if (jok) {
        /* Jacobian structure unchanged; P^{-1} still valid */
        *jcurPtr = SUNFALSE;
        return 0;
    }

    *jcurPtr = SUNTRUE;

    const sunrealtype *yd = N_VGetDeviceArrayPointer_Cuda(y);

    /* Launch psetup kernel: same 2D grid as RHS kernel */
    const dim3 block(16, 8);
    const dim3 grid((pd->ng + block.x - 1) / block.x,
                    (pd->ny + block.y - 1) / block.y);

    psetup_kernel<<<grid, block>>>(yd, gamma, pd->d_Pinv,
                                   pd->ng, pd->ny, pd->ncell);

    cudaError_t e = cudaPeekAtLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "precond: psetup_kernel failed: %s\n",
                cudaGetErrorString(e));
        return -1;
    }

    /* No explicit sync needed: CVODE will call psolve after this,
     * and psolve is on the same stream (0), so ordering is preserved. */
    return 0;
}

/* =========================================================
 * CVODE psolve callback
 *
 * Applies z = P^{-1} r.
 * lr = 1 → left preconditioner (we use left; SPGMR default).
 * lr = 2 → right preconditioner (also works, same math here).
 * ========================================================= */
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

    const int ncell  = pd->ncell;
    const int block  = 256;
    const int grid   = (ncell + block - 1) / block;

    psolve_kernel<<<grid, block>>>(rd, zd, pd->d_Pinv, ncell);

    cudaError_t e = cudaPeekAtLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "precond: psolve_kernel failed: %s\n",
                cudaGetErrorString(e));
        return -1;
    }

    return 0;
}
