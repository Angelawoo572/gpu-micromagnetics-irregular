/*
 * precond.cu — Block-Jacobi preconditioner for spin-wave slit LLG.
 *
 * Easy axis z: c_msk={0,0,1} → only J33 gets anisotropy self-coupling.
 * DMI x:       c_nsk={1,0,0} → no self-coupling contribution.
 *
 * PcUserData must be byte-compatible with UserData in 2d_slit.cu:
 *   offset 0  : void *pd_opaque
 *   offset 8  : sunrealtype *d_hdmag
 *   offset 16 : sunrealtype *d_ymsk
 *   offset 24 : void *demag
 *   offset 32 : int nx, ny, ng, ncell, neq
 *   offset 52 : int screen_col, slit_lo, slit_hi, src_col
 *   offset 68 : double nxx0, nyy0, nzz0
 *   offset 92 : double omega_drive
 */

#include "precond.h"
#include <cvode/cvode.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef C_CHE
#define C_CHE   4.0
#endif
#ifndef C_CHK
#define C_CHK   0.5
#endif
#ifndef C_CHB
#define C_CHB   0.1
#endif
#ifndef C_ALPHA
#define C_ALPHA 0.01
#endif
#ifndef C_CHG
#define C_CHG   1.0
#endif
#ifndef PC_BLOCK_SIZE
#define PC_BLOCK_SIZE 256
#endif

/* ─── Index helpers ─────────────────────────────────────────────────── */
__device__ static inline int pidx_mx(int c, int nc) { return c; }
__device__ static inline int pidx_my(int c, int nc) { return nc + c; }
__device__ static inline int pidx_mz(int c, int nc) { return 2*nc + c; }
__device__ static inline int pwrap_x(int x, int ng) {
    return (x < 0) ? x+ng : (x >= ng ? x-ng : x);
}
__device__ static inline int pwrap_y(int y, int ny) {
    return (y < 0) ? y+ny : (y >= ny ? y-ny : y);
}

/* ─── Opaque types ──────────────────────────────────────────────────── */
struct PrecondData {
    int ng, ny, ncell;
    sunrealtype *d_P;   /* 9*ncell doubles, SoA: P[s*ncell + cell] */
};

typedef struct {
    void        *pd_opaque;
    sunrealtype *d_hdmag;
    sunrealtype *d_ymsk;
    void        *demag;
    int  nx, ny, ng, ncell, neq;
    int  screen_col, slit_lo, slit_hi, src_col;
    double nxx0, nyy0, nzz0;
    double omega_drive;
} PcUserData;

/* ─── build_J_kernel ────────────────────────────────────────────────── */
/*
 * z-axis anisotropy: c_msk={0,0,1}.
 * Self-field contributions to J33: chk*(3m3^2-1) + nzz0 (demag).
 * J11: nxx0 (demag only).
 * J22: nyy0 (demag only).
 * Off-diagonals: from LLG cross product and mh terms.
 */
__global__ static void build_J_kernel(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ h_dmag,
    sunrealtype*       __restrict__ d_P,
    sunrealtype gamma,
    sunrealtype nxx0, sunrealtype nyy0, sunrealtype nzz0,
    int ng, int ny, int ncell)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= ng || gy >= ny) return;

    const int cell = gy*ng + gx;
    const int mx = pidx_mx(cell,ncell);
    const int my = pidx_my(cell,ncell);
    const int mz = pidx_mz(cell,ncell);

    const int xl = pwrap_x(gx-1,ng), xr = pwrap_x(gx+1,ng);
    const int yu = pwrap_y(gy-1,ny), ydn = pwrap_y(gy+1,ny);
    const int lc=gy*ng+xl, rc=gy*ng+xr, uc=yu*ng+gx, dc=ydn*ng+gx;

    const sunrealtype m1 = y[mx], m2 = y[my], m3 = y[mz];

    /* Effective field (same as f_kernel, z-axis anisotropy) */
    const sunrealtype h1 =
        (sunrealtype)C_CHE*(y[pidx_mx(lc,ncell)]+y[pidx_mx(rc,ncell)]
                           +y[pidx_mx(uc,ncell)]+y[pidx_mx(dc,ncell)])
        + (sunrealtype)C_CHB*(y[pidx_mx(lc,ncell)]+y[pidx_mx(rc,ncell)])
        + h_dmag[mx];

    const sunrealtype h2 =
        (sunrealtype)C_CHE*(y[pidx_my(lc,ncell)]+y[pidx_my(rc,ncell)]
                           +y[pidx_my(uc,ncell)]+y[pidx_my(dc,ncell)])
        + h_dmag[my];

    const sunrealtype h3 =
        (sunrealtype)C_CHE*(y[pidx_mz(lc,ncell)]+y[pidx_mz(rc,ncell)]
                           +y[pidx_mz(uc,ncell)]+y[pidx_mz(dc,ncell)])
        + (sunrealtype)C_CHK * m3*(m3*m3 - SUN_RCONST(1.0))
        + h_dmag[mz];

    const sunrealtype mh = m1*h1 + m2*h2 + m3*h3;

    /* Derivative of h w.r.t. self m (only self-terms survive) */
    /* dh1/dm1 = nxx0,  dh2/dm2 = nyy0,  dh3/dm3 = chk*(3m3^2-1)+nzz0 */
    const sunrealtype dh1_dm1 = nxx0;
    const sunrealtype dh2_dm2 = nyy0;
    const sunrealtype dh3_dm3 = (sunrealtype)C_CHK*(SUN_RCONST(3.0)*m3*m3-SUN_RCONST(1.0)) + nzz0;

    /* J = df/dm (self block only) */
    /* J[α][β] = c_chg*(ε_αγβ h_γ + ε_αγδ m_γ dh_δ/dm_β)
     *         + c_alpha*(dh_α/dm_β - dm_h/dm_β * m_α - mh * δ_αβ) */
    /* For brevity: only diagonal approximate is used here — standard for
     * block-Jacobi preconditioners.  The full 3x3 block is built below. */

    /* df1/dm1 */ sunrealtype J11 = (sunrealtype)C_CHG*(m3*SUN_RCONST(0.0)-m2*SUN_RCONST(0.0))
                                   +(sunrealtype)C_ALPHA*(dh1_dm1-(m1*h1+m2*SUN_RCONST(0.0)+m3*SUN_RCONST(0.0))*m1-mh);
    /* df1/dm2 */ sunrealtype J12 = (sunrealtype)C_CHG*(m3*dh2_dm2-SUN_RCONST(0.0))
                                   +(sunrealtype)C_ALPHA*(SUN_RCONST(0.0)-(m1*SUN_RCONST(0.0)+m2*h2)*m1);
    /* df1/dm3 */ sunrealtype J13 = (sunrealtype)C_CHG*(SUN_RCONST(0.0)*h2-SUN_RCONST(0.0)*h3-m2*dh3_dm3)
                                   +(sunrealtype)C_ALPHA*(SUN_RCONST(0.0)-(m3*h3)*m1);  /* simplified */

    /* df2/dm1 */ sunrealtype J21 = (sunrealtype)C_CHG*(SUN_RCONST(0.0)*h3-m3*dh1_dm1)
                                   +(sunrealtype)C_ALPHA*(SUN_RCONST(0.0)-(m1*h1)*m2);
    /* df2/dm2 */ sunrealtype J22 = (sunrealtype)C_CHG*(m1*SUN_RCONST(0.0)-SUN_RCONST(0.0))
                                   +(sunrealtype)C_ALPHA*(dh2_dm2-(m2*h2)*m2-mh);  /* approx */
    /* df2/dm3 */ sunrealtype J23 = (sunrealtype)C_CHG*(m1*dh3_dm3-SUN_RCONST(0.0)*h1)
                                   +(sunrealtype)C_ALPHA*(SUN_RCONST(0.0)-(m3*h3)*m2);

    /* df3/dm1 */ sunrealtype J31 = (sunrealtype)C_CHG*(m2*dh1_dm1-SUN_RCONST(0.0)*h2)
                                   +(sunrealtype)C_ALPHA*(SUN_RCONST(0.0)-(m1*h1)*m3);
    /* df3/dm2 */ sunrealtype J32 = (sunrealtype)C_CHG*(SUN_RCONST(0.0)*h1-m1*dh2_dm2)
                                   +(sunrealtype)C_ALPHA*(SUN_RCONST(0.0)-(m2*h2)*m3);
    /* df3/dm3 */ sunrealtype J33 = (sunrealtype)C_CHG*(SUN_RCONST(0.0))
                                   +(sunrealtype)C_ALPHA*(dh3_dm3-(m3*h3)*m3-mh);

    /* A = I - gamma*J */
    sunrealtype A00 = SUN_RCONST(1.0) - gamma*J11;
    sunrealtype A01 =                 - gamma*J12;
    sunrealtype A02 =                 - gamma*J13;
    sunrealtype A10 =                 - gamma*J21;
    sunrealtype A11 = SUN_RCONST(1.0) - gamma*J22;
    sunrealtype A12 =                 - gamma*J23;
    sunrealtype A20 =                 - gamma*J31;
    sunrealtype A21 =                 - gamma*J32;
    sunrealtype A22 = SUN_RCONST(1.0) - gamma*J33;

    /* Inverse via Cramer's rule */
    sunrealtype det = A00*(A11*A22-A12*A21)
                    - A01*(A10*A22-A12*A20)
                    + A02*(A10*A21-A11*A20);
    sunrealtype inv = (det*det > SUN_RCONST(1e-30)) ?
                      SUN_RCONST(1.0)/det : SUN_RCONST(1.0);

    /* Store P^{-1} in SoA layout: d_P[s*ncell + cell] */
    d_P[0*ncell+cell] =  inv*(A11*A22-A12*A21);
    d_P[1*ncell+cell] = -inv*(A01*A22-A02*A21);
    d_P[2*ncell+cell] =  inv*(A01*A12-A02*A11);
    d_P[3*ncell+cell] = -inv*(A10*A22-A12*A20);
    d_P[4*ncell+cell] =  inv*(A00*A22-A02*A20);
    d_P[5*ncell+cell] = -inv*(A00*A12-A02*A10);
    d_P[6*ncell+cell] =  inv*(A10*A21-A11*A20);
    d_P[7*ncell+cell] = -inv*(A00*A21-A01*A20);
    d_P[8*ncell+cell] =  inv*(A00*A11-A01*A10);
}

/* ─── apply_P_kernel ────────────────────────────────────────────────── */
__global__ static void apply_P_kernel(
    const sunrealtype* __restrict__ d_P,
    const sunrealtype* __restrict__ r,
    sunrealtype*       __restrict__ z,
    const sunrealtype* __restrict__ ymsk,
    int ncell)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= ncell) return;

    const sunrealtype r0 = r[cell];
    const sunrealtype r1 = r[ncell+cell];
    const sunrealtype r2 = r[2*ncell+cell];

    const sunrealtype z0 = d_P[0*ncell+cell]*r0 + d_P[1*ncell+cell]*r1 + d_P[2*ncell+cell]*r2;
    const sunrealtype z1 = d_P[3*ncell+cell]*r0 + d_P[4*ncell+cell]*r1 + d_P[5*ncell+cell]*r2;
    const sunrealtype z2 = d_P[6*ncell+cell]*r0 + d_P[7*ncell+cell]*r1 + d_P[8*ncell+cell]*r2;

    z[cell]         = ymsk[cell]         * z0;
    z[ncell+cell]   = ymsk[ncell+cell]   * z1;
    z[2*ncell+cell] = ymsk[2*ncell+cell] * z2;
}

/* ─── Public API ────────────────────────────────────────────────────── */
PrecondData* Precond_Create(int ng, int ny, int ncell)
{
    PrecondData *pd = (PrecondData*)calloc(1, sizeof(PrecondData));
    if (!pd) { fprintf(stderr, "[Precond] calloc failed\n"); return NULL; }
    pd->ng = ng; pd->ny = ny; pd->ncell = ncell;

    const size_t bytes = (size_t)9 * ncell * sizeof(sunrealtype);
    if (cudaMalloc((void**)&pd->d_P, bytes) != cudaSuccess) {
        fprintf(stderr, "[Precond] cudaMalloc failed\n");
        free(pd); return NULL;
    }
    cudaMemset(pd->d_P, 0, bytes);
    printf("[Precond] Block-Jacobi 3x3 (z-axis aniso): "
           "ncell=%d  %.2f MB\n", ncell, (double)bytes/1e6);
    return pd;
}

void Precond_Destroy(PrecondData *pd)
{
    if (!pd) return;
    if (pd->d_P) cudaFree(pd->d_P);
    free(pd);
}

int PrecondSetup(sunrealtype t, N_Vector y, N_Vector fy,
                 sunbooleantype jok, sunbooleantype* jcurPtr,
                 sunrealtype gamma, void* user_data)
{
    (void)t; (void)fy; (void)jok;
    PcUserData  *ud = (PcUserData*)user_data;
    PrecondData *pd = (PrecondData*)ud->pd_opaque;
    *jcurPtr = SUNTRUE;

    sunrealtype *ydata = N_VGetDeviceArrayPointer_Cuda(y);
    dim3 block(16,8);
    dim3 grid((ud->ng+15)/16, (ud->ny+7)/8);
    build_J_kernel<<<grid, block>>>(
        ydata, ud->d_hdmag, pd->d_P, gamma,
        (sunrealtype)ud->nxx0, (sunrealtype)ud->nyy0, (sunrealtype)ud->nzz0,
        ud->ng, ud->ny, ud->ncell);

    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "[PrecondSetup] kernel failed: %s\n",
                cudaGetErrorString(cudaGetLastError()));
        return -1;
    }
    return 0;
}

int PrecondSolve(sunrealtype t, N_Vector y, N_Vector fy,
                 N_Vector r, N_Vector z,
                 sunrealtype gamma, sunrealtype delta,
                 int lr, void* user_data)
{
    (void)t; (void)y; (void)fy; (void)gamma; (void)delta; (void)lr;
    PcUserData  *ud = (PcUserData*)user_data;
    PrecondData *pd = (PrecondData*)ud->pd_opaque;

    const int b = PC_BLOCK_SIZE;
    const int g = (ud->ncell + b - 1) / b;
    apply_P_kernel<<<g, b>>>(
        pd->d_P,
        N_VGetDeviceArrayPointer_Cuda(r),
        N_VGetDeviceArrayPointer_Cuda(z),
        ud->d_ymsk,
        ud->ncell);

    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "[PrecondSolve] kernel failed: %s\n",
                cudaGetErrorString(cudaGetLastError()));
        return -1;
    }
    return 0;
}
