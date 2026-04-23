/*
 * jtv.cu  —  Analytic Jacobian-times-vector for 2D periodic LLG solver.
 *
 * x-axis anisotropy version, NO DMI (c_chb = 0).
 *
 * ── Physical model ──────────────────────────────────────────────────────────
 *
 * Constants (must match 2d_fft.cu and precond.cu):
 *   c_msk = {1,0,0}, c_nsk = {1,0,0}
 *   c_chk = 1.0, c_che = 4.0, c_alpha = 0.2, c_chg = 1.0
 *   c_cha = 0.0, c_chb = 0.0  (NO DMI)
 *
 * Effective field (c_chb=0, c_cha=0, c_msk={1,0,0}):
 *   H1 = c_che*(m1_L+m1_R+m1_U+m1_D) + c_chk*m1
 *   H2 = c_che*(m2_L+m2_R+m2_U+m2_D)
 *   H3 = c_che*(m3_L+m3_R+m3_U+m3_D)
 *
 * ── Jacobian derivation ──────────────────────────────────────────────────────
 *
 * LLG RHS:
 *   f1 = c_chg*(m3*H2 - m2*H3) + c_alpha*(H1 - (m·H)*m1)
 *   f2 = c_chg*(m1*H3 - m3*H1) + c_alpha*(H2 - (m·H)*m2)
 *   f3 = c_chg*(m2*H1 - m1*H2) + c_alpha*(H3 - (m·H)*m3)
 *
 * Perturbation dH from perturbation v (y frozen):
 *   dH1 = c_che*(v1_L+v1_R+v1_U+v1_D) + c_chk*v1   ← self-coupling in H1
 *   dH2 = c_che*(v2_L+v2_R+v2_U+v2_D)
 *   dH3 = c_che*(v3_L+v3_R+v3_U+v3_D)
 *
 * Note: c_chb=0 → no DMI term in dH1.
 *       c_chk*v1 appears in dH1 (x-axis anisotropy self-coupling).
 *       Compare z-axis version: c_chk*v3 appeared in dH3.
 *
 * d(m·H) = (v·H) + (m·dH)  ≡  dmh
 *
 * Jv components:
 *   (Jv)1 = c_chg*(v3*H2 + m3*dH2 - v2*H3 - m2*dH3)
 *            + c_alpha*(dH1 - dmh*m1 - mh*v1)
 *
 *   (Jv)2 = c_chg*(v1*H3 + m1*dH3 - v3*H1 - m3*dH1)
 *            + c_alpha*(dH2 - dmh*m2 - mh*v2)
 *
 *   (Jv)3 = c_chg*(v2*H1 + m2*dH1 - v1*H2 - m1*dH2)
 *            + c_alpha*(dH3 - dmh*m3 - mh*v3)
 *
 * JtvUserData mirrors UserData from 2d_fft.cu exactly:
 *   offset 0:  PrecondData*
 *   offset 8:  DemagData*
 *   offset 16: sunrealtype* d_hdmag
 *   offset 24: int nx
 *   offset 28: int ny
 *   offset 32: int ng
 *   offset 36: int ncell
 *   offset 40: int neq
 */

#include "jtv.h"
#include <cvode/cvode.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

/* Constants — must match 2d_fft.cu and precond.cu */
__constant__ sunrealtype jc_msk[3] = {1.0, 0.0, 0.0};
__constant__ sunrealtype jc_nsk[3] = {1.0, 0.0, 0.0};
__constant__ sunrealtype jc_chk    = 1.0;
__constant__ sunrealtype jc_che    = 4.0;
__constant__ sunrealtype jc_alpha  = 0.2;
__constant__ sunrealtype jc_chg    = 1.0;
__constant__ sunrealtype jc_cha    = 0.0;
__constant__ sunrealtype jc_chb    = 0.0;   /* NO DMI */

__device__ static inline int jidx_mx(int c,int nc){return c;}
__device__ static inline int jidx_my(int c,int nc){return nc+c;}
__device__ static inline int jidx_mz(int c,int nc){return 2*nc+c;}
__device__ static inline int jwrap_x(int x,int ng){return (x<0)?(x+ng):((x>=ng)?(x-ng):x);}
__device__ static inline int jwrap_y(int y,int ny){return (y<0)?(y+ny):((y>=ny)?(y-ny):y);}

__global__ static void jtv_kernel(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ v,
    sunrealtype*       __restrict__ Jv,
    int ng, int ny, int ncell)
{
  const int gx=blockIdx.x*blockDim.x+threadIdx.x;
  const int gy=blockIdx.y*blockDim.y+threadIdx.y;
  if(gx>=ng||gy>=ny) return;

  const int cell=gy*ng+gx;
  const int mx=jidx_mx(cell,ncell), my=jidx_my(cell,ncell), mz=jidx_mz(cell,ncell);

  const int xl=jwrap_x(gx-1,ng), xr=jwrap_x(gx+1,ng);
  const int yu=jwrap_y(gy-1,ny), ydn=jwrap_y(gy+1,ny);
  const int lc=gy*ng+xl, rc=gy*ng+xr, uc=yu*ng+gx, dc=ydn*ng+gx;

  const sunrealtype m1=y[mx], m2=y[my], m3=y[mz];
  const sunrealtype v1=v[mx], v2=v[my], v3=v[mz];

  /* neighbor v values */
  const sunrealtype v1L=v[jidx_mx(lc,ncell)], v1R=v[jidx_mx(rc,ncell)];
  const sunrealtype v1U=v[jidx_mx(uc,ncell)], v1D=v[jidx_mx(dc,ncell)];
  const sunrealtype v2L=v[jidx_my(lc,ncell)], v2R=v[jidx_my(rc,ncell)];
  const sunrealtype v2U=v[jidx_my(uc,ncell)], v2D=v[jidx_my(dc,ncell)];
  const sunrealtype v3L=v[jidx_mz(lc,ncell)], v3R=v[jidx_mz(rc,ncell)];
  const sunrealtype v3U=v[jidx_mz(uc,ncell)], v3D=v[jidx_mz(dc,ncell)];

  /*
   * Effective field H at frozen y.
   * c_chb=0: DMI term absent.
   * c_msk={1,0,0}: anisotropy only in H1, via c_chk*m1.
   * c_cha=0: no bias.
   *
   * H1 = c_che*(m1_L+m1_R+m1_U+m1_D) + c_chk*m1
   * H2 = c_che*(m2_L+m2_R+m2_U+m2_D)
   * H3 = c_che*(m3_L+m3_R+m3_U+m3_D)
   */
  const sunrealtype h1 =
      jc_che*(y[jidx_mx(lc,ncell)]+y[jidx_mx(rc,ncell)]+
              y[jidx_mx(uc,ncell)]+y[jidx_mx(dc,ncell)]) +
      jc_chk*m1;

  const sunrealtype h2 =
      jc_che*(y[jidx_my(lc,ncell)]+y[jidx_my(rc,ncell)]+
              y[jidx_my(uc,ncell)]+y[jidx_my(dc,ncell)]);

  const sunrealtype h3 =
      jc_che*(y[jidx_mz(lc,ncell)]+y[jidx_mz(rc,ncell)]+
              y[jidx_mz(uc,ncell)]+y[jidx_mz(dc,ncell)]);

  const sunrealtype mh = m1*h1 + m2*h2 + m3*h3;

  /*
   * Linearized field dH from perturbation v.
   * dH1 = c_che*(v1 neighbors) + c_chk*v1   [anisotropy self-coupling]
   * dH2 = c_che*(v2 neighbors)
   * dH3 = c_che*(v3 neighbors)
   *
   * c_chb=0 → no (v1_L+v1_R) DMI term in dH1.
   */
  const sunrealtype dh1 = jc_che*(v1L+v1R+v1U+v1D) + jc_chk*v1;
  const sunrealtype dh2 = jc_che*(v2L+v2R+v2U+v2D);
  const sunrealtype dh3 = jc_che*(v3L+v3R+v3U+v3D);

  /* d(m·H) = v·H + m·dH */
  const sunrealtype dmh = (v1*h1+v2*h2+v3*h3) + (m1*dh1+m2*dh2+m3*dh3);

  Jv[mx] = jc_chg*(v3*h2+m3*dh2-v2*h3-m2*dh3) + jc_alpha*(dh1-dmh*m1-mh*v1);
  Jv[my] = jc_chg*(v1*h3+m1*dh3-v3*h1-m3*dh1) + jc_alpha*(dh2-dmh*m2-mh*v2);
  Jv[mz] = jc_chg*(v2*h1+m2*dh1-v1*h2-m1*dh2) + jc_alpha*(dh3-dmh*m3-mh*v3);
}

typedef struct {
  void *pd_opaque;
  void *demag_opaque;
  void *d_hdmag_opaque;
  int nx, ny, ng, ncell, neq;
} JtvUserData;

int JtvProduct(N_Vector v, N_Vector Jv, sunrealtype t,
               N_Vector y, N_Vector fy, void* user_data, N_Vector tmp)
{
  (void)t;(void)fy;(void)tmp;
  const JtvUserData* ud=(const JtvUserData*)user_data;
  const sunrealtype* yd =N_VGetDeviceArrayPointer_Cuda(y);
  const sunrealtype* vd =N_VGetDeviceArrayPointer_Cuda(v);
  sunrealtype*       Jvd=N_VGetDeviceArrayPointer_Cuda(Jv);

  const dim3 block(16,8);
  const dim3 grid((ud->ng+15)/16,(ud->ny+7)/8);
  jtv_kernel<<<grid,block>>>(yd,vd,Jvd,ud->ng,ud->ny,ud->ncell);

  if(cudaPeekAtLastError()!=cudaSuccess){
    fprintf(stderr,"jtv_kernel failed: %s\n",cudaGetErrorString(cudaGetLastError()));
    return -1;
  }
  return 0;
}
