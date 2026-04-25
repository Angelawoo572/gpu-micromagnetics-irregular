/*
 * precond.cu — Block-Jacobi 3×3 preconditioner, ymsk version.
 *
 * Per cell we approximate the local Jacobian J_local = ∂f_cell/∂m_cell
 * (neighbors held fixed) and form
 *
 *     A = I − γ J_local          (3×3)
 *
 * then store A⁻¹ in device memory.  PrecondSolve is one dense 3×3
 * mat-vec per cell.
 *
 * ─── ymsk handling ──────────────────────────────────────────────────
 *   apply_P_kernel:  z[mα] = ymsk[mα] * (P · r)[mα]
 *
 * That's it.  No `if (active[…])` branches in build_J_kernel; it just
 * computes the local J for every cell.  Hole cells get garbage J / P⁻¹
 * blocks, but the apply_P mask drops their z to 0 — so any noise in
 * the GMRES Krylov subspace at hole cells stays at 0 throughout the
 * iteration.  Build is unconditional, simple, fast.
 *
 * ─── LLG form ───────────────────────────────────────────────────────
 *   dm/dt = chg (m × h) + α ( h − (m·h) m )
 *
 * Standard simplified form; |m|≈1 is enforced by normalize_m_kernel
 * in 2d_fft.cu's f() with +0.01 regularization.
 *
 * ─── What contributes to J_local ─────────────────────────────────────
 *   h_cell = h_exchange(neighbors) + h_DMI(neighbors)
 *          + h_anisotropy(m1) + h_demag(convolution)
 *
 *   Self derivatives (only terms depending on this cell's m):
 *     ∂h1/∂m1 = c_chk + Nxx(0)·strength      ← anisotropy + demag-self
 *     ∂h2/∂m2 =          Nyy(0)·strength
 *     ∂h3/∂m3 =          Nzz(0)·strength
 *     off-diagonals zero (c_msk diag, N(0) diag by 4-fold symmetry).
 *
 *   Shorthand:  k1 = c_chk + Nxx0,  k2 = Nyy0,  k3 = Nzz0.
 *
 * ─── Closed-form J ──────────────────────────────────────────────────
 *   Let mh = m·h, e_β = h_β + m_β k_β = ∂(m·h)/∂m_β.
 *
 *     J[0][0] = α (k1 − e1 m1 − mh)
 *     J[0][1] = chg (m3 k2 − h3) − α e2 m1
 *     J[0][2] = chg (h2 − m2 k3) − α e3 m1
 *     J[1][0] = chg (h3 − m3 k1) − α e1 m2
 *     J[1][1] = α (k2 − e2 m2 − mh)
 *     J[1][2] = chg (m1 k3 − h1) − α e3 m2
 *     J[2][0] = chg (m2 k1 − h2) − α e1 m3
 *     J[2][1] = chg (h1 − m1 k2) − α e2 m3
 *     J[2][2] = α (k3 − e3 m3 − mh)
 */

#include "precond.h"

#include <cvode/cvode.h>
#include <nvector/nvector_cuda.h>
#include <sundials/sundials_types.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

/* ─── Material constants (mirror 2d_fft.cu values) ──────────────────── */
__constant__ static sunrealtype pc_chk   = SUN_RCONST(4.0);
__constant__ static sunrealtype pc_che   = SUN_RCONST(4.0);
__constant__ static sunrealtype pc_alpha = SUN_RCONST(0.2);
__constant__ static sunrealtype pc_chg   = SUN_RCONST(1.0);
__constant__ static sunrealtype pc_cha   = SUN_RCONST(0.0);
__constant__ static sunrealtype pc_chb   = SUN_RCONST(0.3);
/* Anisotropy axis fixed at {1,0,0} (x); DMI direction fixed at {1,0,0}.
 * These are baked into the kernel rather than loaded from constant mem
 * to save some fetches — change here if the axes ever change. */

/* ─── Indexing helpers ──────────────────────────────────────────────── */
__device__ static inline int pidx_mx(int c, int nc) { return c; }
__device__ static inline int pidx_my(int c, int nc) { return nc + c; }
__device__ static inline int pidx_mz(int c, int nc) { return 2*nc + c; }
__device__ static inline int pwrap_x(int x, int ng) {
    return (x < 0) ? (x + ng) : ((x >= ng) ? (x - ng) : x);
}
__device__ static inline int pwrap_y(int y, int ny) {
    return (y < 0) ? (y + ny) : ((y >= ny) ? (y - ny) : y);
}

/* ─── Opaque types ──────────────────────────────────────────────────── */
struct PrecondData {
    int ng, ny, ncell;
    sunrealtype *d_P;    /* 9*ncell doubles; SoA: P[s*ncell + cell] */
};

/*
 * PcUserData: byte-compatible mirror of UserData in 2d_fft.cu.
 *   offset 0  : void* pd_opaque
 *   offset 8  : void* demag_opaque
 *   offset 16 : sunrealtype *d_hdmag
 *   offset 24 : sunrealtype *d_ymsk
 *   offset 32 : int nx, ny, ng, ncell, neq   (5*4 = 20 bytes)
 *   offset 56 : double nxx0, nyy0, nzz0
 */
typedef struct {
    void        *pd_opaque;
    void        *demag_opaque;
    sunrealtype *d_hdmag;
    sunrealtype *d_ymsk;
    int nx, ny, ng, ncell, neq;
    double nxx0, nyy0, nzz0;
} PcUserData;

/* ─── build_J_kernel ────────────────────────────────────────────────── */
/*
 * One thread per cell.  Computes the 3×3 local J unconditionally and
 * stores its inverse.  Hole-cell results are junk but irrelevant —
 * apply_P_kernel drops them to 0 via the ymsk multiply.
 *
 * Hole-cell neighbor reads return 0 (m frozen there), so for active
 * cells the h calculation is correct without any branching.
 */
__global__ static void build_J_kernel(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ h_dmag,   /* may be non-null; SoA 3*ncell */
    sunrealtype*       __restrict__ d_P,
    sunrealtype gamma,
    sunrealtype nxx0, sunrealtype nyy0, sunrealtype nzz0,
    int ng, int ny, int ncell)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= ng || gy >= ny) return;

    const int cell = gy * ng + gx;
    const int mx = pidx_mx(cell, ncell);
    const int my = pidx_my(cell, ncell);
    const int mz = pidx_mz(cell, ncell);

    const int xl = pwrap_x(gx - 1, ng);
    const int xr = pwrap_x(gx + 1, ng);
    const int yu = pwrap_y(gy - 1, ny);
    const int ydn= pwrap_y(gy + 1, ny);

    const int lc = gy  * ng + xl;
    const int rc = gy  * ng + xr;
    const int uc = yu  * ng + gx;
    const int dc = ydn * ng + gx;

    const sunrealtype m1 = y[mx];
    const sunrealtype m2 = y[my];
    const sunrealtype m3 = y[mz];

    /* Total h at this cell — anisotropy on h1 only (c_msk={1,0,0}),
     * DMI on h1 only (c_nsk={1,0,0}). */
    sunrealtype h1 =
    pc_che * (y[pidx_mx(lc,ncell)] + y[pidx_mx(rc,ncell)] +
              y[pidx_mx(uc,ncell)] + y[pidx_mx(dc,ncell)])
  + pc_chk * m1 * (m1 * m1 - m1);

    sunrealtype h2 =
        pc_che * (y[pidx_my(lc,ncell)] + y[pidx_my(rc,ncell)] +
                y[pidx_my(uc,ncell)] + y[pidx_my(dc,ncell)])
    + pc_chk * m2 * (m2 * m2 - m2);

    sunrealtype h3 =
        pc_che * (y[pidx_mz(lc,ncell)] + y[pidx_mz(rc,ncell)] +
                y[pidx_mz(uc,ncell)] + y[pidx_mz(dc,ncell)])
    + pc_chk * m3 * (m3 * m3 - m3);

    if (h_dmag) {
        h1 += h_dmag[mx];
        h2 += h_dmag[my];
        h3 += h_dmag[mz];
    }

    /* Self-coupling diagonal:  ∂h_α/∂m_α at this cell. */
    // const sunrealtype k1 = pc_chk + nxx0;     /* anisotropy + demag-self */
    // const sunrealtype k2 = nyy0;
    // const sunrealtype k3 = nzz0;
    const sunrealtype k1 =
    pc_chk * (SUN_RCONST(3.0) * m1 * m1 - SUN_RCONST(2.0) * m1) + nxx0;

    const sunrealtype k2 =
        pc_chk * (SUN_RCONST(3.0) * m2 * m2 - SUN_RCONST(2.0) * m2) + nyy0;

    const sunrealtype k3 =
        pc_chk * (SUN_RCONST(3.0) * m3 * m3 - SUN_RCONST(2.0) * m3) + nzz0;

    const sunrealtype e1 = h1 + m1 * k1;
    const sunrealtype e2 = h2 + m2 * k2;
    const sunrealtype e3 = h3 + m3 * k3;
    const sunrealtype mh = m1 * h1 + m2 * h2 + m3 * h3;

    const sunrealtype J00 = pc_alpha * (k1 - e1 * m1 - mh);
    const sunrealtype J01 = pc_chg * (m3 * k2 - h3) - pc_alpha * e2 * m1;
    const sunrealtype J02 = pc_chg * (h2 - m2 * k3) - pc_alpha * e3 * m1;

    const sunrealtype J10 = pc_chg * (h3 - m3 * k1) - pc_alpha * e1 * m2;
    const sunrealtype J11 = pc_alpha * (k2 - e2 * m2 - mh);
    const sunrealtype J12 = pc_chg * (m1 * k3 - h1) - pc_alpha * e3 * m2;

    const sunrealtype J20 = pc_chg * (m2 * k1 - h2) - pc_alpha * e1 * m3;
    const sunrealtype J21 = pc_chg * (h1 - m1 * k2) - pc_alpha * e2 * m3;
    const sunrealtype J22 = pc_alpha * (k3 - e3 * m3 - mh);

    /* A = I − γ J */
    const sunrealtype A00 = SUN_RCONST(1.0) - gamma * J00;
    const sunrealtype A01 =                 - gamma * J01;
    const sunrealtype A02 =                 - gamma * J02;
    const sunrealtype A10 =                 - gamma * J10;
    const sunrealtype A11 = SUN_RCONST(1.0) - gamma * J11;
    const sunrealtype A12 =                 - gamma * J12;
    const sunrealtype A20 =                 - gamma * J20;
    const sunrealtype A21 =                 - gamma * J21;
    const sunrealtype A22 = SUN_RCONST(1.0) - gamma * J22;

    /* 3×3 inverse via adjugate / det. */
    const sunrealtype c00 = A11 * A22 - A12 * A21;
    const sunrealtype c10 = A12 * A20 - A10 * A22;
    const sunrealtype c20 = A10 * A21 - A11 * A20;
    const sunrealtype det = A00 * c00 + A01 * c10 + A02 * c20;

    /* Fall back to identity if det is pathological. */
    if (!(fabs((double)det) > 1e-30)) {
        d_P[0*ncell + cell] = SUN_RCONST(1.0);
        d_P[1*ncell + cell] = SUN_RCONST(0.0);
        d_P[2*ncell + cell] = SUN_RCONST(0.0);
        d_P[3*ncell + cell] = SUN_RCONST(0.0);
        d_P[4*ncell + cell] = SUN_RCONST(1.0);
        d_P[5*ncell + cell] = SUN_RCONST(0.0);
        d_P[6*ncell + cell] = SUN_RCONST(0.0);
        d_P[7*ncell + cell] = SUN_RCONST(0.0);
        d_P[8*ncell + cell] = SUN_RCONST(1.0);
        return;
    }

    const sunrealtype inv_det = SUN_RCONST(1.0) / det;

    /* A⁻¹[i][j] = (adj A)[i][j] / det,  adj A = transpose(cofactors). */
    const sunrealtype Pi00 = (A11 * A22 - A12 * A21) * inv_det;
    const sunrealtype Pi01 = (A02 * A21 - A01 * A22) * inv_det;
    const sunrealtype Pi02 = (A01 * A12 - A02 * A11) * inv_det;
    const sunrealtype Pi10 = (A12 * A20 - A10 * A22) * inv_det;
    const sunrealtype Pi11 = (A00 * A22 - A02 * A20) * inv_det;
    const sunrealtype Pi12 = (A02 * A10 - A00 * A12) * inv_det;
    const sunrealtype Pi20 = (A10 * A21 - A11 * A20) * inv_det;
    const sunrealtype Pi21 = (A01 * A20 - A00 * A21) * inv_det;
    const sunrealtype Pi22 = (A00 * A11 - A01 * A10) * inv_det;

    d_P[0*ncell + cell] = Pi00;
    d_P[1*ncell + cell] = Pi01;
    d_P[2*ncell + cell] = Pi02;
    d_P[3*ncell + cell] = Pi10;
    d_P[4*ncell + cell] = Pi11;
    d_P[5*ncell + cell] = Pi12;
    d_P[6*ncell + cell] = Pi20;
    d_P[7*ncell + cell] = Pi21;
    d_P[8*ncell + cell] = Pi22;
}

/* ─── apply_P_kernel: z = ymsk * (P⁻¹ r) ───────────────────────────── */
__global__ static void apply_P_kernel(
    const sunrealtype* __restrict__ P,
    const sunrealtype* __restrict__ r,
    const sunrealtype* __restrict__ ymsk,
    sunrealtype*       __restrict__ z,
    int ncell)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= ncell) return;

    const sunrealtype r0 = r[cell];
    const sunrealtype r1 = r[ncell + cell];
    const sunrealtype r2 = r[2*ncell + cell];

    const sunrealtype z0 =
        P[0*ncell + cell] * r0 +
        P[1*ncell + cell] * r1 +
        P[2*ncell + cell] * r2;
    const sunrealtype z1 =
        P[3*ncell + cell] * r0 +
        P[4*ncell + cell] * r1 +
        P[5*ncell + cell] * r2;
    const sunrealtype z2 =
        P[6*ncell + cell] * r0 +
        P[7*ncell + cell] * r1 +
        P[8*ncell + cell] * r2;

    /* Mask drops hole-cell output to 0. */
    z[cell]           = ymsk[cell]           * z0;
    z[ncell + cell]   = ymsk[ncell + cell]   * z1;
    z[2*ncell + cell] = ymsk[2*ncell + cell] * z2;
}

/* ─── Create/Destroy ────────────────────────────────────────────────── */
PrecondData* Precond_Create(int ng, int ny, int ncell)
{
    PrecondData *pd = (PrecondData*)calloc(1, sizeof(PrecondData));
    if (!pd) {
        fprintf(stderr, "[Precond] calloc failed\n");
        return NULL;
    }
    pd->ng    = ng;
    pd->ny    = ny;
    pd->ncell = ncell;
    pd->d_P   = NULL;

    const size_t bytes = (size_t)9 * ncell * sizeof(sunrealtype);
    cudaError_t err = cudaMalloc((void**)&pd->d_P, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[Precond] cudaMalloc failed: %s\n",
                cudaGetErrorString(err));
        free(pd);
        return NULL;
    }
    cudaMemset(pd->d_P, 0, bytes);

    printf("[Precond] Block-Jacobi 3x3: ncell=%d, device mem = %.2f MB\n",
           ncell, (double)bytes / 1e6);
    return pd;
}

void Precond_Destroy(PrecondData *pd)
{
    if (!pd) return;
    if (pd->d_P) cudaFree(pd->d_P);
    free(pd);
}

/* ─── PrecondSetup (rebuild A⁻¹ for given gamma and y) ──────────────── */
int PrecondSetup(sunrealtype t, N_Vector y, N_Vector fy,
                 sunbooleantype jok, sunbooleantype* jcurPtr,
                 sunrealtype gamma, void* user_data)
{
    (void)t; (void)fy; (void)jok;

    PcUserData  *ud = (PcUserData*)user_data;
    PrecondData *pd = (PrecondData*)ud->pd_opaque;

    *jcurPtr = SUNTRUE;

    sunrealtype *ydata = N_VGetDeviceArrayPointer_Cuda(y);

    dim3 block(16, 8);
    dim3 grid((ud->ng + block.x - 1) / block.x,
              (ud->ny + block.y - 1) / block.y);

    build_J_kernel<<<grid, block>>>(
        ydata,
        ud->d_hdmag,
        pd->d_P,
        gamma,
        (sunrealtype)ud->nxx0,
        (sunrealtype)ud->nyy0,
        (sunrealtype)ud->nzz0,
        ud->ng, ud->ny, ud->ncell);

    cudaError_t cuerr = cudaPeekAtLastError();
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "[PrecondSetup] kernel launch failed: %s\n",
                cudaGetErrorString(cuerr));
        return -1;
    }
    return 0;
}

/* ─── PrecondSolve (apply cached A⁻¹ block per cell, mask output) ──── */
int PrecondSolve(sunrealtype t, N_Vector y, N_Vector fy,
                 N_Vector r, N_Vector z,
                 sunrealtype gamma, sunrealtype delta,
                 int lr, void* user_data)
{
    (void)t; (void)y; (void)fy; (void)gamma; (void)delta; (void)lr;

    PcUserData  *ud = (PcUserData*)user_data;
    PrecondData *pd = (PrecondData*)ud->pd_opaque;

    sunrealtype *rdata = N_VGetDeviceArrayPointer_Cuda(r);
    sunrealtype *zdata = N_VGetDeviceArrayPointer_Cuda(z);

    const int b = 256;
    const int g = (ud->ncell + b - 1) / b;
    apply_P_kernel<<<g, b>>>(pd->d_P, rdata, ud->d_ymsk, zdata, ud->ncell);

    cudaError_t cuerr = cudaPeekAtLastError();
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "[PrecondSolve] kernel launch failed: %s\n",
                cudaGetErrorString(cuerr));
        return -1;
    }
    return 0;
}
