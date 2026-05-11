/*
 * jtv.cu — Analytic Jv for spin-wave slit LLG.
 *
 * Physics: easy axis along z (c_msk={0,0,1}), DMI along x (c_nsk={1,0,0}).
 * This differs from i2 (x-axis anisotropy) — here anisotropy is on h3.
 *
 * Demag NOT included in Jv (would need second FFT per GMRES iter).
 * Hole cells masked by ymsk → Jv = 0 there.
 *
 * JtvUserData layout must match UserData in 2d_slit.cu byte-for-byte:
 *   offset 0  : void *pd_opaque
 *   offset 8  : sunrealtype *d_hdmag
 *   offset 16 : sunrealtype *d_ymsk
 *   offset 24 : DemagData *demag
 *   offset 32 : int nx, ny, ng, ncell, neq
 *   offset 52 : int screen_col, slit_lo, slit_hi, src_col
 *   offset 68 : double nxx0, nyy0, nzz0
 *   offset 92 : double omega_drive
 */

#include "jtv.h"
#include <cvode/cvode.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

/* Constants must match 2d_slit.cu -D flags */
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

__constant__ sunrealtype jc_che   = C_CHE;
__constant__ sunrealtype jc_chk   = C_CHK;
__constant__ sunrealtype jc_chb   = C_CHB;
__constant__ sunrealtype jc_alpha = C_ALPHA;
__constant__ sunrealtype jc_chg   = C_CHG;

__device__ static inline int jidx_mx(int c, int nc) { return c; }
__device__ static inline int jidx_my(int c, int nc) { return nc + c; }
__device__ static inline int jidx_mz(int c, int nc) { return 2*nc + c; }
__device__ static inline int jwrap_x(int x, int ng) {
    return (x < 0) ? x+ng : (x >= ng ? x-ng : x);
}
__device__ static inline int jwrap_y(int y, int ny) {
    return (y < 0) ? y+ny : (y >= ny ? y-ny : y);
}

/*
 * Analytic Jv kernel.
 *
 * Easy axis z: c_msk={0,0,1} → only h3 gets anisotropy self-coupling.
 * DMI x:       c_nsk={1,0,0} → only h1 gets DMI from x-neighbors.
 *
 * h1 = che*(y1L+y1R+y1U+y1D) + chb*(y1L+y1R)
 * h2 = che*(y2L+y2R+y2U+y2D)
 * h3 = che*(y3L+y3R+y3U+y3D) + chk*m3*(m3²-1)
 *
 * dh1 = (che+chb)*(v1L+v1R) + che*(v1U+v1D)
 * dh2 = che*(v2L+v2R+v2U+v2D)
 * dh3 = che*(v3L+v3R+v3U+v3D) + chk*(3m3²-1)*v3
 */
__global__ static void jtv_kernel(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ v,
    const sunrealtype* __restrict__ ymsk,
    sunrealtype*       __restrict__ Jv,
    int ng, int ny, int ncell)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= ng || gy >= ny) return;

    const int cell = gy*ng + gx;
    const int mx = jidx_mx(cell,ncell);
    const int my = jidx_my(cell,ncell);
    const int mz = jidx_mz(cell,ncell);

    const int xl = jwrap_x(gx-1,ng), xr = jwrap_x(gx+1,ng);
    const int yu = jwrap_y(gy-1,ny), ydn = jwrap_y(gy+1,ny);
    const int lc = gy*ng+xl, rc = gy*ng+xr;
    const int uc = yu*ng+gx, dc = ydn*ng+gx;

    const sunrealtype m1 = y[mx], m2 = y[my], m3 = y[mz];
    const sunrealtype v1 = v[mx], v2 = v[my], v3 = v[mz];

    /* y neighbors */
    const sunrealtype y1L=y[jidx_mx(lc,ncell)], y1R=y[jidx_mx(rc,ncell)];
    const sunrealtype y1U=y[jidx_mx(uc,ncell)], y1D=y[jidx_mx(dc,ncell)];
    const sunrealtype y2L=y[jidx_my(lc,ncell)], y2R=y[jidx_my(rc,ncell)];
    const sunrealtype y2U=y[jidx_my(uc,ncell)], y2D=y[jidx_my(dc,ncell)];
    const sunrealtype y3L=y[jidx_mz(lc,ncell)], y3R=y[jidx_mz(rc,ncell)];
    const sunrealtype y3U=y[jidx_mz(uc,ncell)], y3D=y[jidx_mz(dc,ncell)];

    /* v neighbors */
    const sunrealtype v1L=v[jidx_mx(lc,ncell)], v1R=v[jidx_mx(rc,ncell)];
    const sunrealtype v1U=v[jidx_mx(uc,ncell)], v1D=v[jidx_mx(dc,ncell)];
    const sunrealtype v2L=v[jidx_my(lc,ncell)], v2R=v[jidx_my(rc,ncell)];
    const sunrealtype v2U=v[jidx_my(uc,ncell)], v2D=v[jidx_my(dc,ncell)];
    const sunrealtype v3L=v[jidx_mz(lc,ncell)], v3R=v[jidx_mz(rc,ncell)];
    const sunrealtype v3U=v[jidx_mz(uc,ncell)], v3D=v[jidx_mz(dc,ncell)];

    /* Effective field at y */
    const sunrealtype h1 = jc_che*(y1L+y1R+y1U+y1D) + jc_chb*(y1L+y1R);
    const sunrealtype h2 = jc_che*(y2L+y2R+y2U+y2D);
    const sunrealtype h3 = jc_che*(y3L+y3R+y3U+y3D)
                         + jc_chk * m3 * (m3*m3 - SUN_RCONST(1.0));

    const sunrealtype mh = m1*h1 + m2*h2 + m3*h3;

    /* Derivative of h w.r.t. v */
    const sunrealtype dh1 = (jc_che+jc_chb)*(v1L+v1R) + jc_che*(v1U+v1D);
    const sunrealtype dh2 = jc_che*(v2L+v2R+v2U+v2D);
    /* k3 = d/dv3 [chk * m3*(m3^2-1)] = chk*(3m3^2-1)*v3 */
    const sunrealtype k3  = jc_chk * (SUN_RCONST(3.0)*m3*m3 - SUN_RCONST(1.0));
    const sunrealtype dh3 = jc_che*(v3L+v3R+v3U+v3D) + k3*v3;

    const sunrealtype dmh = (v1*h1+v2*h2+v3*h3) + (m1*dh1+m2*dh2+m3*dh3);

    Jv[mx] = ymsk[mx]*(jc_chg*(v3*h2+m3*dh2-v2*h3-m2*dh3)
                      +jc_alpha*(dh1-dmh*m1-mh*v1));
    Jv[my] = ymsk[my]*(jc_chg*(v1*h3+m1*dh3-v3*h1-m3*dh1)
                      +jc_alpha*(dh2-dmh*m2-mh*v2));
    Jv[mz] = ymsk[mz]*(jc_chg*(v2*h1+m2*dh1-v1*h2-m1*dh2)
                      +jc_alpha*(dh3-dmh*m3-mh*v3));
}

/* UserData layout mirror for jtv.cu — matches 2d_slit.cu exactly */
typedef struct {
    void           *pd_opaque;
    sunrealtype    *d_hdmag;
    sunrealtype    *d_ymsk;
    void           *demag;
    int   nx, ny, ng, ncell, neq;
    int   screen_col, slit_lo, slit_hi, src_col;
    double nxx0, nyy0, nzz0;
    double omega_drive;
} JtvUserData;

int JtvProduct(N_Vector v, N_Vector Jv, sunrealtype t,
               N_Vector y, N_Vector fy, void *user_data, N_Vector tmp)
{
    (void)t; (void)fy; (void)tmp;
    const JtvUserData *ud = (const JtvUserData*)user_data;

    const dim3 block(16, 8);
    const dim3 grid((ud->ng + 15)/16, (ud->ny + 7)/8);

    jtv_kernel<<<grid, block>>>(
        N_VGetDeviceArrayPointer_Cuda(y),
        N_VGetDeviceArrayPointer_Cuda(v),
        ud->d_ymsk,
        N_VGetDeviceArrayPointer_Cuda(Jv),
        ud->ng, ud->ny, ud->ncell);

    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "jtv_kernel failed: %s\n",
                cudaGetErrorString(cudaGetLastError()));
        return -1;
    }
    return 0;
}
