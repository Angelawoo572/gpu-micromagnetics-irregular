/*
 * jtv.cu -- Analytic Jacobian-times-vector, x-axis anisotropy, with demag
 *
 * Constants match 2d_fft.cu:
 *   c_chk=1.0, c_che=4.0, c_alpha=0.2, c_chg=1.0, c_cha=0.0, c_chb=0.3
 *
 * H1 = (c_che+c_chb)*(m1_L+m1_R) + c_che*(m1_U+m1_D) + c_chk*m1
 * H2 = c_che*(m2 neighbors)
 * H3 = c_che*(m3 neighbors)
 *
 * dH1 = (c_che+c_chb)*(v1_L+v1_R) + c_che*(v1_U+v1_D) + c_chk*v1
 * dH2 = c_che*(v2 neighbors)
 * dH3 = c_che*(v3 neighbors)
 *
 * Demag (h_d linear in m): h_d(m+eps*v) = h_d(m) + eps*h_d(v)
 *   Compute hdv = h_d(v) via Demag_Apply, then add demag Jacobian to Jv.
 */

#include "jtv.h"
#include "demag_fft.h"
#include <cvode/cvode.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__constant__ sunrealtype jc_chk   = 1.0;
__constant__ sunrealtype jc_che   = 4.0;
__constant__ sunrealtype jc_alpha = 0.2;
__constant__ sunrealtype jc_chg   = 1.0;
__constant__ sunrealtype jc_cha   = 0.0;
__constant__ sunrealtype jc_chb   = 0.3;

__device__ static inline int jidx_mx(int c,int nc){return c;}
__device__ static inline int jidx_my(int c,int nc){return nc+c;}
__device__ static inline int jidx_mz(int c,int nc){return 2*nc+c;}
__device__ static inline int jwrap_x(int x,int ng){return(x<0)?(x+ng):((x>=ng)?(x-ng):x);}
__device__ static inline int jwrap_y(int y,int ny){return(y<0)?(y+ny):((y>=ny)?(y-ny):y);}

/* Kernel 1: exchange + anisotropy Jv */
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
  const int mx=jidx_mx(cell,ncell),my=jidx_my(cell,ncell),mz=jidx_mz(cell,ncell);

  const int xl=jwrap_x(gx-1,ng),xr=jwrap_x(gx+1,ng);
  const int yu=jwrap_y(gy-1,ny),ydn=jwrap_y(gy+1,ny);
  const int lc=gy*ng+xl,rc=gy*ng+xr,uc=yu*ng+gx,dc=ydn*ng+gx;

  const sunrealtype m1=y[mx],m2=y[my],m3=y[mz];
  const sunrealtype v1=v[mx],v2=v[my],v3=v[mz];

  const sunrealtype v1L=v[jidx_mx(lc,ncell)],v1R=v[jidx_mx(rc,ncell)];
  const sunrealtype v1U=v[jidx_mx(uc,ncell)],v1D=v[jidx_mx(dc,ncell)];
  const sunrealtype v2L=v[jidx_my(lc,ncell)],v2R=v[jidx_my(rc,ncell)];
  const sunrealtype v2U=v[jidx_my(uc,ncell)],v2D=v[jidx_my(dc,ncell)];
  const sunrealtype v3L=v[jidx_mz(lc,ncell)],v3R=v[jidx_mz(rc,ncell)];
  const sunrealtype v3U=v[jidx_mz(uc,ncell)],v3D=v[jidx_mz(dc,ncell)];

  const sunrealtype che_chb = jc_che + jc_chb;

  const sunrealtype h1 =
      che_chb*(y[jidx_mx(lc,ncell)]+y[jidx_mx(rc,ncell)]) +
      jc_che *(y[jidx_mx(uc,ncell)]+y[jidx_mx(dc,ncell)]) +
      jc_chk*m1 + jc_cha;

  const sunrealtype h2 =
      jc_che*(y[jidx_my(lc,ncell)]+y[jidx_my(rc,ncell)]+
              y[jidx_my(uc,ncell)]+y[jidx_my(dc,ncell)]);

  const sunrealtype h3 =
      jc_che*(y[jidx_mz(lc,ncell)]+y[jidx_mz(rc,ncell)]+
              y[jidx_mz(uc,ncell)]+y[jidx_mz(dc,ncell)]);

  const sunrealtype mh = m1*h1+m2*h2+m3*h3;

  const sunrealtype dh1 = che_chb*(v1L+v1R)+jc_che*(v1U+v1D)+jc_chk*v1;
  const sunrealtype dh2 = jc_che*(v2L+v2R+v2U+v2D);
  const sunrealtype dh3 = jc_che*(v3L+v3R+v3U+v3D);

  const sunrealtype dmh = (v1*h1+v2*h2+v3*h3)+(m1*dh1+m2*dh2+m3*dh3);

  Jv[mx] = jc_chg*(v3*h2+m3*dh2-v2*h3-m2*dh3)+jc_alpha*(dh1-dmh*m1-mh*v1);
  Jv[my] = jc_chg*(v1*h3+m1*dh3-v3*h1-m3*dh1)+jc_alpha*(dh2-dmh*m2-mh*v2);
  Jv[mz] = jc_chg*(v2*h1+m2*dh1-v1*h2-m1*dh2)+jc_alpha*(dh3-dmh*m3-mh*v3);
}

/* Kernel 2: demag contribution (accumulate onto Jv with +=) */
__global__ static void jtv_demag_kernel(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ v,
    const sunrealtype* __restrict__ hd,
    const sunrealtype* __restrict__ hdv,
    sunrealtype*       __restrict__ Jv,
    int ncell)
{
  const int c = blockIdx.x*blockDim.x + threadIdx.x;
  if (c >= ncell) return;

  const sunrealtype m1=y[c],          m2=y[ncell+c],          m3=y[2*ncell+c];
  const sunrealtype v1=v[c],          v2=v[ncell+c],          v3=v[2*ncell+c];
  const sunrealtype hd1=hd[c],        hd2=hd[ncell+c],        hd3=hd[2*ncell+c];
  const sunrealtype hdv1=hdv[c],      hdv2=hdv[ncell+c],      hdv3=hdv[2*ncell+c];

  const sunrealtype mh  = m1*hd1 + m2*hd2 + m3*hd3;
  const sunrealtype dmh = (v1*hd1+v2*hd2+v3*hd3) + (m1*hdv1+m2*hdv2+m3*hdv3);

  Jv[c]         += jc_chg*(v3*hd2+m3*hdv2-v2*hd3-m2*hdv3)
                 + jc_alpha*(hdv1-dmh*m1-mh*v1);
  Jv[ncell+c]   += jc_chg*(v1*hd3+m1*hdv3-v3*hd1-m3*hdv1)
                 + jc_alpha*(hdv2-dmh*m2-mh*v2);
  Jv[2*ncell+c] += jc_chg*(v2*hd1+m2*hdv1-v1*hd2-m1*hdv2)
                 + jc_alpha*(hdv3-dmh*m3-mh*v3);
}

/* JtvUserData byte-compatible with UserData in 2d_fft.cu:
 *   0  : PrecondData*    pd
 *   8  : DemagData*      demag
 *   16 : sunrealtype*    d_hdmag    (h_demag(y))
 *   24 : sunrealtype*    d_hdmag_v  (h_demag(v))   <-- NEW
 *   32 : int nx
 *   36 : int ny
 *   40 : int ng
 *   44 : int ncell
 *   48 : int neq
 */
typedef struct {
  void *pd_opaque;
  void *demag_opaque;
  void *d_hdmag_opaque;
  void *d_hdmag_v_opaque;
  int nx, ny, ng, ncell, neq;
} JtvUserData;

int JtvProduct(N_Vector v, N_Vector Jv, sunrealtype t,
               N_Vector y, N_Vector fy, void* user_data, N_Vector tmp)
{
  (void)t; (void)fy; (void)tmp;
  const JtvUserData* ud = (const JtvUserData*)user_data;

  sunrealtype* yd  = N_VGetDeviceArrayPointer_Cuda(y);
  sunrealtype* vd  = N_VGetDeviceArrayPointer_Cuda(v);
  sunrealtype* Jvd = N_VGetDeviceArrayPointer_Cuda(Jv);

  /* 1) exchange + anisotropy */
  const dim3 block(16,8);
  const dim3 grid((ud->ng+15)/16, (ud->ny+7)/8);
  jtv_kernel<<<grid,block>>>(yd, vd, Jvd, ud->ng, ud->ny, ud->ncell);

  /* 2) demag (only if active) */
  if (ud->demag_opaque != NULL && ud->d_hdmag_v_opaque != NULL) {
    sunrealtype* h_dv       = (sunrealtype*)ud->d_hdmag_v_opaque;
    const sunrealtype* h_d  = (const sunrealtype*)ud->d_hdmag_opaque;

    cudaMemsetAsync(h_dv, 0, (size_t)3*ud->ncell*sizeof(sunrealtype), 0);
    Demag_Apply((DemagData*)ud->demag_opaque,
                (const double*)vd, (double*)h_dv);

    const int b=256, g=(ud->ncell+255)/256;
    jtv_demag_kernel<<<g,b>>>(yd, vd, h_d, h_dv, Jvd, ud->ncell);
  }

  if (cudaPeekAtLastError() != cudaSuccess) {
    fprintf(stderr,"jtv failed: %s\n",cudaGetErrorString(cudaGetLastError()));
    return -1;
  }
  return 0;
}
