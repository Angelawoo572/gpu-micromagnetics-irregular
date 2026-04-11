/*
 * jtv.cu  —  Analytic Jacobian-times-vector for the 2D periodic LLG solver.
 *
 * =========================================================================
 * Why this exists
 * =========================================================================
 * CVODE's default Jv approximation:
 *
 *   Jv ≈ [f(y + ε·v) - f(y)] / ε
 *
 * costs one full RHS evaluation per GMRES iteration.  With 20 468 GMRES
 * iterations observed in profiling, that is 20 468 extra launches of
 * f_kernel_group_soa_periodic (~85 µs each) = ~1.74 s spent purely on
 * finite-difference overhead, contributing nothing to accuracy.
 *
 * This file replaces that with an analytic kernel that:
 *   - calls f() zero times
 *   - has no ε-tuning issue
 *   - runs in the same time as one RHS evaluation
 *   - is registered via CVodeSetJacTimes(cvode_mem, NULL, JtvProduct)
 *
 * =========================================================================
 * Full analytic derivation
 * =========================================================================
 *
 * LLG RHS (cell i, SoA layout):
 *   f1 = c_chg*(m3*h2 - m2*h3) + c_alpha*(h1 - mh*m1)
 *   f2 = c_chg*(m1*h3 - m3*h1) + c_alpha*(h2 - mh*m2)
 *   f3 = c_chg*(m2*h1 - m1*h2) + c_alpha*(h3 - mh*m3)
 * where mh = m1*h1 + m2*h2 + m3*h3.
 *
 * Effective field:
 *   h1 = c_che*(mxL+mxR+mxU+mxD) + c_msk[0]*(c_chk*m3+c_cha)
 *        + c_chb*c_nsk[0]*(mxL+mxR)
 *   h2 = c_che*(myL+myR+myU+myD) + c_msk[1]*(c_chk*m3+c_cha)
 *        + c_chb*c_nsk[1]*(myL+myR)
 *   h3 = c_che*(mzL+mzR+mzU+mzD) + c_msk[2]*(c_chk*m3+c_cha)
 *        + c_chb*c_nsk[2]*(mzL+mzR)
 *
 * Substituting our constants (c_msk=[0,0,1], c_nsk=[1,0,0], c_chk=1,
 * c_cha=0, c_msk[1]=0, c_nsk[1]=0, c_nsk[2]=0):
 *   h1 = (c_che+c_chb)*(mxL+mxR) + c_che*(mxU+mxD)
 *   h2 = c_che*(myL+myR+myU+myD)
 *   h3 = c_che*(mzL+mzR+mzU+mzD) + c_chk*m3
 *      = c_che*(mzL+mzR+mzU+mzD) + m3
 *
 * Differentiate w.r.t. perturbation v (y fixed):
 *   dh1 = (c_che+c_chb)*(v1L+v1R) + c_che*(v1U+v1D)
 *   dh2 = c_che*(v2L+v2R+v2U+v2D)
 *   dh3 = c_che*(v3L+v3R+v3U+v3D) + v3      (from c_chk*v3 self term)
 *
 * dmh = v1*h1 + v2*h2 + v3*h3    (v·H, H is current field at y)
 *      + m1*dh1 + m2*dh2 + m3*dh3  (m·dH)
 *
 * (Jv)_i components:
 *   jv1 = c_chg*(v3*h2 + m3*dh2 - v2*h3 - m2*dh3)
 *        + c_alpha*(dh1 - dmh*m1 - mh*v1)
 *
 *   jv2 = c_chg*(v1*h3 + m1*dh3 - v3*h1 - m3*dh1)
 *        + c_alpha*(dh2 - dmh*m2 - mh*v2)
 *
 *   jv3 = c_chg*(v2*h1 + m2*dh1 - v1*h2 - m1*dh2)
 *        + c_alpha*(dh3 - dmh*m3 - mh*v3)
 *
 * This is computed in one CUDA kernel with the same stencil structure
 * as the RHS kernel.  No extra f() call, no ε.
 *
 * =========================================================================
 * Connection to parallel-systems theory (15-418 / PMPP)
 * =========================================================================
 * From the PMPP slides perspective, analytic Jv is "work elimination":
 * we remove 20 K redundant stencil kernel launches (the FD evaluations)
 * by exploiting problem structure.  This is the GPU analog of Halide's
 * "don't recompute what you can derive analytically."
 *
 * From the SUNDIALS GPU slides perspective, this is the recommended
 * pattern: provide a problem-specific jtimes callback implemented as a
 * GPU kernel, so the entire Newton-Krylov inner loop stays GPU-resident.
 */

#include "jtv.h"

#include <cvode/cvode.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

/* =========================================================
 * Physical constants — must match 2d_p.cu exactly.
 * Stored in __constant__ memory for broadcast efficiency.
 * ========================================================= */
__constant__ sunrealtype jc_msk[3]  = {0.0, 0.0, 1.0};
__constant__ sunrealtype jc_nsk[3]  = {1.0, 0.0, 0.0};
__constant__ sunrealtype jc_chk     = 1.0;
__constant__ sunrealtype jc_che     = 4.0;
__constant__ sunrealtype jc_alpha   = 0.2;
__constant__ sunrealtype jc_chg     = 1.0;
__constant__ sunrealtype jc_cha     = 0.0;
__constant__ sunrealtype jc_chb     = 0.3;

/* =========================================================
 * Index / wrap helpers (SoA, same as 2d_p.cu)
 * ========================================================= */
__device__ static inline int jidx_mx(int c, int nc) { return c; }
__device__ static inline int jidx_my(int c, int nc) { return nc + c; }
__device__ static inline int jidx_mz(int c, int nc) { return 2*nc + c; }

__device__ static inline int jwrap_x(int x, int ng) {
    return (x < 0) ? (x+ng) : ((x >= ng) ? (x-ng) : x);
}
__device__ static inline int jwrap_y(int y, int ny) {
    return (y < 0) ? (y+ny) : ((y >= ny) ? (y-ny) : y);
}

/* =========================================================
 * jtv_kernel
 *
 * Computes (J·v)_i analytically for each cell i.
 *
 * Inputs:
 *   y  — current state (device pointer, SoA)
 *   v  — Krylov direction vector (device pointer, SoA)
 * Output:
 *   Jv — result J(y)·v (device pointer, SoA)
 *
 * One thread per cell, same 2-D launch as f_kernel.
 * ========================================================= */
__global__ static void jtv_kernel(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ v,
    sunrealtype*       __restrict__ Jv,
    int ng, int ny, int ncell)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= ng || gy >= ny) return;

    const int cell = gy * ng + gx;

    /* ---- self indices ---- */
    const int mx = jidx_mx(cell, ncell);
    const int my = jidx_my(cell, ncell);
    const int mz = jidx_mz(cell, ncell);

    /* ---- neighbor cell indices ---- */
    const int xl   = jwrap_x(gx-1, ng),  xr   = jwrap_x(gx+1, ng);
    const int yu   = jwrap_y(gy-1, ny),  ydn  = jwrap_y(gy+1, ny);
    const int lc   = gy*ng + xl,          rc   = gy*ng + xr;
    const int uc   = yu*ng + gx,          dc   = ydn*ng + gx;

    /* ---- current magnetization m ---- */
    const sunrealtype m1 = y[mx];
    const sunrealtype m2 = y[my];
    const sunrealtype m3 = y[mz];

    /* ---- current perturbation v (self + neighbors) ---- */
    const sunrealtype v1 = v[mx];
    const sunrealtype v2 = v[my];
    const sunrealtype v3 = v[mz];

    /* neighbor v components */
    const sunrealtype v1L = v[jidx_mx(lc, ncell)],  v1R = v[jidx_mx(rc, ncell)];
    const sunrealtype v1U = v[jidx_mx(uc, ncell)],  v1D = v[jidx_mx(dc, ncell)];
    const sunrealtype v2L = v[jidx_my(lc, ncell)],  v2R = v[jidx_my(rc, ncell)];
    const sunrealtype v2U = v[jidx_my(uc, ncell)],  v2D = v[jidx_my(dc, ncell)];
    const sunrealtype v3L = v[jidx_mz(lc, ncell)],  v3R = v[jidx_mz(rc, ncell)];
    const sunrealtype v3U = v[jidx_mz(uc, ncell)],  v3D = v[jidx_mz(dc, ncell)];

    /* ---- effective field H at current y ---- */
    /*
     * h1 = (c_che + c_chb)*(mxL+mxR) + c_che*(mxU+mxD)   [nsk[0]=1]
     * h2 = c_che*(myL+myR+myU+myD)                         [nsk[1]=0]
     * h3 = c_che*(mzL+mzR+mzU+mzD) + m3                   [msk[2]=1,chk=1]
     */
    const sunrealtype che_chb = jc_che + jc_chb;   /* precomputed */

    const sunrealtype h1 =
        che_chb * (y[jidx_mx(lc,ncell)] + y[jidx_mx(rc,ncell)]) +
        jc_che  * (y[jidx_mx(uc,ncell)] + y[jidx_mx(dc,ncell)]);

    const sunrealtype h2 =
        jc_che * (y[jidx_my(lc,ncell)] + y[jidx_my(rc,ncell)] +
                  y[jidx_my(uc,ncell)] + y[jidx_my(dc,ncell)]);

    const sunrealtype h3 =
        jc_che * (y[jidx_mz(lc,ncell)] + y[jidx_mz(rc,ncell)] +
                  y[jidx_mz(uc,ncell)] + y[jidx_mz(dc,ncell)])
        + m3;       /* c_msk[2]*c_chk*m3 = 1*1*m3 */

    const sunrealtype mh = m1*h1 + m2*h2 + m3*h3;

    /* ---- dH: derivative of H w.r.t. the perturbation v ---- */
    /*
     * dh1 = (c_che+c_chb)*(v1L+v1R) + c_che*(v1U+v1D)
     * dh2 = c_che*(v2L+v2R+v2U+v2D)
     * dh3 = c_che*(v3L+v3R+v3U+v3D) + v3   (self: c_msk[2]*c_chk*v3 = v3)
     */
    const sunrealtype dh1 =
        che_chb * (v1L + v1R) +
        jc_che  * (v1U + v1D);

    const sunrealtype dh2 =
        jc_che * (v2L + v2R + v2U + v2D);

    const sunrealtype dh3 =
        jc_che * (v3L + v3R + v3U + v3D) + v3;

    /* ---- dmh = v·H + m·dH ---- */
    const sunrealtype dmh = (v1*h1 + v2*h2 + v3*h3)
                           + (m1*dh1 + m2*dh2 + m3*dh3);

    /* ---- (Jv)_i = D_m[f] · v  (full analytic formula) ---- */
    /*
     * jv1 = c_chg*(v3*h2 + m3*dh2 - v2*h3 - m2*dh3)
     *      + c_alpha*(dh1 - dmh*m1 - mh*v1)
     *
     * jv2 = c_chg*(v1*h3 + m1*dh3 - v3*h1 - m3*dh1)
     *      + c_alpha*(dh2 - dmh*m2 - mh*v2)
     *
     * jv3 = c_chg*(v2*h1 + m2*dh1 - v1*h2 - m1*dh2)
     *      + c_alpha*(dh3 - dmh*m3 - mh*v3)
     */
    Jv[mx] = jc_chg * (v3*h2 + m3*dh2 - v2*h3 - m2*dh3)
            + jc_alpha * (dh1 - dmh*m1 - mh*v1);

    Jv[my] = jc_chg * (v1*h3 + m1*dh3 - v3*h1 - m3*dh1)
            + jc_alpha * (dh2 - dmh*m2 - mh*v2);

    Jv[mz] = jc_chg * (v2*h1 + m2*dh1 - v1*h2 - m1*dh2)
            + jc_alpha * (dh3 - dmh*m3 - mh*v3);
}

/* =========================================================
 * UserData forward declaration
 * (mirrors the struct in 2d_p.cu; only ng/ny/ncell are needed)
 * ========================================================= */
typedef struct {
    void *pd_opaque;   /* PrecondData* — not used here */
    int   nx, ny, ng, ncell, neq;
} JtvUserData;

/* =========================================================
 * JtvProduct  —  CVODE jtimes callback
 * ========================================================= */
int JtvProduct(N_Vector v,  N_Vector Jv,
               sunrealtype t,
               N_Vector y,  N_Vector fy,
               void *user_data,
               N_Vector tmp)
{
    (void)t; (void)fy; (void)tmp;

    const JtvUserData *ud = (const JtvUserData*)user_data;

    const sunrealtype *yd  = N_VGetDeviceArrayPointer_Cuda(y);
    const sunrealtype *vd  = N_VGetDeviceArrayPointer_Cuda(v);
    sunrealtype       *Jvd = N_VGetDeviceArrayPointer_Cuda(Jv);

    /* Same 2-D launch geometry as f_kernel */
    const dim3 block(16, 8);
    const dim3 grid((ud->ng + block.x - 1) / block.x,
                    (ud->ny + block.y - 1) / block.y);

    jtv_kernel<<<grid, block>>>(yd, vd, Jvd,
                                ud->ng, ud->ny, ud->ncell);

    const cudaError_t e = cudaPeekAtLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "jtv_kernel launch failed: %s\n",
                cudaGetErrorString(e));
        return -1;
    }
    /* No explicit sync: CVODE's SPGMR will issue the next
     * CUDA operation on stream 0, ensuring ordering. */
    return 0;
}
