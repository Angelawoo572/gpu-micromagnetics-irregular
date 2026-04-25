/*
 * precond.cu — Block-Jacobi 3×3 preconditioner for CVODE Newton/GMRES.
 *
 * Per cell we approximate the local Jacobian J_local = ∂f_cell/∂m_cell
 * (neighbors held fixed) and form
 *
 *     A = I − γ J_local          (3×3)
 *
 * then store A⁻¹ in device memory.  PrecondSolve is one dense 3×3
 * mat-vec per cell.
 *
 * ─── LLG form matched here ──────────────────────────────────────────
 *   dm/dt = chg (m × h) + alpha ( h − (m·h) m )
 *
 * This is the standard simplified form assuming |m|=1.  The assumption is
 * enforced by normalize_m_kernel in 2d_fft.cu's f(), which projects y onto
 * the unit sphere at the top of every RHS evaluation.
 *
 * ─── What contributes to J_local ─────────────────────────────────────
 *   h_cell = h_exchange(neighbors) + h_DMI(neighbors)
 *          + h_anisotropy(m1) + h_demag(convolution)
 *
 *   Only terms that depend on THIS cell's m appear in ∂h/∂m_self:
 *     ∂h1/∂m1 = c_chk  + Nxx(0)·strength
 *     ∂h2/∂m2 =           Nyy(0)·strength
 *     ∂h3/∂m3 =           Nzz(0)·strength
 *     off-diagonals zero (c_msk components off, N(0) diagonal by symmetry).
 *
 *   Shorthand:  k1 = c_chk + Nxx0,  k2 = Nyy0,  k3 = Nzz0.
 *
 * ─── Derivatives (closed form) ───────────────────────────────────────
 *   Let mm = |m|², mh = m·h, e_β = h_β + m_β k_β = ∂(m·h)/∂m_β.
 *
 *   Precession part d/dm_β [chg (m×h)_α]:
 *     d (m3 h2 − m2 h3)/dm  for α=1   (matches 2d_fft.cu's sign convention)
 *     d (m1 h3 − m3 h1)/dm  for α=2
 *     d (m2 h1 − m1 h2)/dm  for α=3
 *
 *   Damping part d/dm_β [alpha (mm h_α − mh m_α)]:
 *     = alpha (2 m_β h_α + mm k_α δ_{αβ} − e_β m_α − mh δ_{αβ})
 *
 *   The NEW contribution vs the old simplified form is the `2 m_β h_α`
 *   term from d|m|²/dm; this is what removes the m·h-driven feedback.
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
/* Anisotropy mask fixed at {1,0,0}; DMI mask fixed at {1,0,0}.
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
 * PcUserData: mirror of UserData in 2d_fft.cu (byte-compatible layout).
 *   offset 0  : PrecondData*
 *   offset 8  : DemagData*
 *   offset 16 : sunrealtype *d_hdmag
 *   offset 24 : int nx, ny, ng, ncell, neq   (5*4 = 20 bytes)
 *   offset 48 : double nxx0, nyy0, nzz0      (after 4-byte pad)
 */
typedef struct {
    void        *pd_opaque;
    void        *demag_opaque;
    sunrealtype *d_hdmag;
    int nx, ny, ng, ncell, neq;
    double nxx0, nyy0, nzz0;
    unsigned char* h_active;
    unsigned char* d_active;
    int* h_active_ids;
    int* d_active_ids;
    int  n_active;
    int* h_inactive_ids;
    int* d_inactive_ids;
    int  n_inactive;
} PcUserData;

/* ─── build_J_kernel ────────────────────────────────────────────────── */
/*
 * One thread per cell. Reads self + 4 neighbors (for h — derivatives
 * only need self quantities) plus h_dmag[mx/my/mz] for this cell,
 * assembles the 3×3 local J (|m|-preserving form), forms A = I − γJ,
 * stores A⁻¹.
 */
__global__ static void build_J_kernel(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ h_dmag,   /* may be non-null; SoA 3*ncell */
    sunrealtype*       __restrict__ d_P,
    sunrealtype gamma,
    sunrealtype nxx0, sunrealtype nyy0, sunrealtype nzz0,
    const unsigned char* __restrict__ active,
    int ng, int ny, int ncell)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= ng || gy >= ny) return;

    const int cell = gy * ng + gx;
    const int mx = pidx_mx(cell, ncell);
    const int my = pidx_my(cell, ncell);
    const int mz = pidx_mz(cell, ncell);

    if (active && !active[cell]) {
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

    const int xl = pwrap_x(gx - 1, ng);
    const int xr = pwrap_x(gx + 1, ng);
    const int yu = pwrap_y(gy - 1, ny);
    const int yd = pwrap_y(gy + 1, ny);

    const int lc = gy * ng + xl;
    const int rc = gy * ng + xr;
    const int uc = yu * ng + gx;
    const int dc = yd * ng + gx;

    const sunrealtype m1 = y[mx];
    const sunrealtype m2 = y[my];
    const sunrealtype m3 = y[mz];

    /* Total h at this cell (matches 2d_fft.cu's f_kernel_unified exactly).
     * Anisotropy: c_msk = {1,0,0} → only h1 gets (c_chk*m1 + c_cha).
     * DMI:        c_nsk = {1,0,0} → only h1 gets c_chb*(lx + rx). */
    const sunrealtype y1L = (!active || active[lc]) ? y[pidx_mx(lc,ncell)] : SUN_RCONST(0.0);
    const sunrealtype y1R = (!active || active[rc]) ? y[pidx_mx(rc,ncell)] : SUN_RCONST(0.0);
    const sunrealtype y1U = (!active || active[uc]) ? y[pidx_mx(uc,ncell)] : SUN_RCONST(0.0);
    const sunrealtype y1D = (!active || active[dc]) ? y[pidx_mx(dc,ncell)] : SUN_RCONST(0.0);
    const sunrealtype y2L = (!active || active[lc]) ? y[pidx_my(lc,ncell)] : SUN_RCONST(0.0);
    const sunrealtype y2R = (!active || active[rc]) ? y[pidx_my(rc,ncell)] : SUN_RCONST(0.0);
    const sunrealtype y2U = (!active || active[uc]) ? y[pidx_my(uc,ncell)] : SUN_RCONST(0.0);
    const sunrealtype y2D = (!active || active[dc]) ? y[pidx_my(dc,ncell)] : SUN_RCONST(0.0);
    const sunrealtype y3L = (!active || active[lc]) ? y[pidx_mz(lc,ncell)] : SUN_RCONST(0.0);
    const sunrealtype y3R = (!active || active[rc]) ? y[pidx_mz(rc,ncell)] : SUN_RCONST(0.0);
    const sunrealtype y3U = (!active || active[uc]) ? y[pidx_mz(uc,ncell)] : SUN_RCONST(0.0);
    const sunrealtype y3D = (!active || active[dc]) ? y[pidx_mz(dc,ncell)] : SUN_RCONST(0.0);

    sunrealtype h1 = pc_che * (y1L + y1R + y1U + y1D)
      + (pc_chk * m1 + pc_cha) + pc_chb * (y1L + y1R);
    sunrealtype h2 = pc_che * (y2L + y2R + y2U + y2D);
    sunrealtype h3 = pc_che * (y3L + y3R + y3U + y3D);

    if (h_dmag) {
        h1 += h_dmag[mx];
        h2 += h_dmag[my];
        h3 += h_dmag[mz];
    }

    /* Self-coupling diagonal:  ∂h_α/∂m_α at this cell. */
    const sunrealtype k1 = pc_chk + nxx0;     /* anisotropy + demag-self */
    const sunrealtype k2 = nyy0;
    const sunrealtype k3 = nzz0;

    const sunrealtype e1 = h1 + m1 * k1;      /* = ∂(m·h)/∂m_1 */
    const sunrealtype e2 = h2 + m2 * k2;
    const sunrealtype e3 = h3 + m3 * k3;
    const sunrealtype mh = m1 * h1 + m2 * h2 + m3 * h3;

    /* Local 3×3 J for standard LLG: dm/dt = chg(m×h) + α(h − (m·h)m).
     * |m|=1 is enforced by normalize_m_kernel in f(), so we can use the
     * simplified Jacobian (no |m|² terms needed). */
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

    /* A⁻¹[i][j] = (adj A)[i][j] / det,  adj A = transpose(cofactors).
     * Standard closed form for 3×3. */
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

/* ─── apply_P_kernel: z = P⁻¹ r (3×3 block per cell) ────────────────── */
__global__ static void apply_P_kernel(
    const sunrealtype* __restrict__ P,
    const sunrealtype* __restrict__ r,
    sunrealtype*       __restrict__ z,
    int ncell)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= ncell) return;

    const sunrealtype r0 = r[cell];
    const sunrealtype r1 = r[ncell + cell];
    const sunrealtype r2 = r[2*ncell + cell];

    z[cell] =
        P[0*ncell + cell] * r0 +
        P[1*ncell + cell] * r1 +
        P[2*ncell + cell] * r2;
    z[ncell + cell] =
        P[3*ncell + cell] * r0 +
        P[4*ncell + cell] * r1 +
        P[5*ncell + cell] * r2;
    z[2*ncell + cell] =
        P[6*ncell + cell] * r0 +
        P[7*ncell + cell] * r1 +
        P[8*ncell + cell] * r2;
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

    printf("[Precond] Block-Jacobi 3x3: ncell=%d, "
           "device mem = %.2f MB\n",
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

    /* Always rebuild: the Jacobian depends on both y AND gamma, and
     * gamma changes across Newton failures / step-size adjustments.
     * jcurPtr tells CVODE the Jacobian data is current. */
    *jcurPtr = SUNTRUE;

    sunrealtype *ydata = N_VGetDeviceArrayPointer_Cuda(y);

    dim3 block(16, 8);
    dim3 grid((ud->ng + block.x - 1) / block.x,
              (ud->ny + block.y - 1) / block.y);

    build_J_kernel<<<grid, block>>>(
        ydata,
        ud->d_hdmag,                     /* may be non-null but zero-filled */
        pd->d_P,
        gamma,
        (sunrealtype)ud->nxx0,
        (sunrealtype)ud->nyy0,
        (sunrealtype)ud->nzz0,
        ud->d_active,
        ud->ng, ud->ny, ud->ncell);

    cudaError_t cuerr = cudaPeekAtLastError();
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "[PrecondSetup] kernel launch failed: %s\n",
                cudaGetErrorString(cuerr));
        return -1;
    }
    return 0;
}

/* ─── PrecondSolve (apply cached A⁻¹ block per cell) ────────────────── */
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
    apply_P_kernel<<<g, b>>>(pd->d_P, rdata, zdata, ud->ncell);

    cudaError_t cuerr = cudaPeekAtLastError();
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "[PrecondSolve] kernel launch failed: %s\n",
                cudaGetErrorString(cuerr));
        return -1;
    }
    return 0;
}
