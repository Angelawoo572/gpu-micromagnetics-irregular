/*
 * precond.cu  — 3×3 Block-Diagonal Jacobi Preconditioner
 * x-axis anisotropy, NO DMI (c_chb = 0).
 *
 * ── Constants (must match 2d_fft.cu and jtv.cu) ─────────────────────────────
 *   c_msk={1,0,0}, c_chk=1.0, c_che=4.0, c_alpha=0.2, c_chg=1.0
 *   c_cha=0.0, c_chb=0.0 (no DMI)
 *
 * ── Effective field (c_chb=0, c_cha=0, c_msk={1,0,0}) ───────────────────────
 *   H1 = c_che*(m1_L+m1_R+m1_U+m1_D) + c_chk*m1
 *   H2 = c_che*(m2_L+m2_R+m2_U+m2_D)
 *   H3 = c_che*(m3_L+m3_R+m3_U+m3_D)
 *
 * ── Jacobian block (self-coupling only, neighbors frozen) ────────────────────
 *
 * Let:
 *   mh = m1*H1 + m2*H2 + m3*H3
 *
 * Self-coupling derivatives (∂H_i/∂m_j at fixed neighbors):
 *   ∂H1/∂m1 = c_chk    (anisotropy self)
 *   ∂H2/∂m2 = 0
 *   ∂H3/∂m3 = 0
 *   all other ∂H_i/∂m_j = 0
 *
 * Therefore:
 *   d1 ≡ ∂(m·H)/∂m1 = H1 + m1*c_chk    ← x-axis: c_chk on d1
 *   d2 ≡ ∂(m·H)/∂m2 = H2
 *   d3 ≡ ∂(m·H)/∂m3 = H3               ← z-axis version had H3+c_chk*m3 here
 *
 * Jacobian rows (∂f_row/∂m_col):
 *
 *   J[0][0] = c_alpha*(-d1*m1 - mh)               [∂f1/∂m1, includes ∂H1/∂m1=c_chk via d1]
 *   J[0][1] = -c_chg*H3 - c_alpha*d2*m1
 *   J[0][2] =  c_chg*H2 - c_alpha*d3*m1            [no gyro cross from ∂H3/∂m3: that's 0]
 *
 *   J[1][0] =  c_chg*H3 - c_alpha*d1*m2
 *   J[1][1] = c_alpha*(-d2*m2 - mh)
 *   J[1][2] = -c_chg*H1 - c_alpha*d3*m2            [no gyro cross]
 *
 *   J[2][0] = -c_chg*H2 - c_alpha*d1*m3
 *   J[2][1] =  c_chg*H1 - c_alpha*d2*m3
 *   J[2][2] = c_alpha*(-d3*m3 - mh)                [no c_chk self term: that's in d1 now]
 *
 * Compare z-axis version had:
 *   J[0][2] = c_chg*(H2-m2) - c_alpha*d3*m1    ← gyro correction from ∂H3/∂m3
 *   J[1][2] = c_chg*(m1-H1) - c_alpha*d3*m2
 *   J[2][2] = c_alpha*(c_chk - d3*m3 - mh)     ← c_chk self in [2][2]
 *
 * In x-axis version those become simpler (no ∂H3/∂m3 anisotropy contribution).
 *
 * ── Two-kernel design (unchanged from v2) ───────────────────────────────────
 *   Kernel 1 (build_J_kernel):    reads y, computes J. Called only on jok=FALSE.
 *   Kernel 2 (build_Pinv_kernel): reads d_J + gamma, computes P^{-1}. Always called.
 *   This ensures P^{-1} = (I - gamma*J)^{-1} is always consistent with current gamma.
 */

#include "precond.h"
#include <cvode/cvode.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Constants — must match 2d_fft.cu and jtv.cu */
__constant__ sunrealtype pc_msk[3] = {1.0, 0.0, 0.0};
__constant__ sunrealtype pc_nsk[3] = {1.0, 0.0, 0.0};
__constant__ sunrealtype pc_chk    = 1.0;
__constant__ sunrealtype pc_che    = 4.0;
__constant__ sunrealtype pc_alpha  = 0.2;
__constant__ sunrealtype pc_chg    = 1.0;
__constant__ sunrealtype pc_cha    = 0.0;
__constant__ sunrealtype pc_chb    = 0.0;   /* NO DMI */

__device__ static inline int pidx_mx(int c,int nc){return c;}
__device__ static inline int pidx_my(int c,int nc){return nc+c;}
__device__ static inline int pidx_mz(int c,int nc){return 2*nc+c;}
__device__ static inline int pwrap_x(int x,int ng){return (x<0)?(x+ng):((x>=ng)?(x-ng):x);}
__device__ static inline int pwrap_y(int y,int ny){return (y<0)?(y+ny):((y>=ny)?(y-ny):y);}

/*
 * Kernel 1: build_J_kernel
 *
 * Computes the analytic 3×3 self-coupling Jacobian block per cell.
 * Called only when jok=SUNFALSE (y changed significantly).
 *
 * x-axis anisotropy changes:
 *   d1 = H1 + c_chk*m1   (was d1=H1 in z-axis)
 *   d3 = H3               (was d3=H3+c_chk*m3 in z-axis)
 *   J[0][2], J[1][2], J[2][2]: no anisotropy gyro cross terms.
 */
__global__ static void build_J_kernel(
    const sunrealtype* __restrict__ y,
    sunrealtype*       __restrict__ d_J,
    int ng, int ny, int ncell)
{
  const int gx=blockIdx.x*blockDim.x+threadIdx.x;
  const int gy=blockIdx.y*blockDim.y+threadIdx.y;
  if(gx>=ng||gy>=ny) return;

  const int cell=gy*ng+gx;
  const sunrealtype m1=y[pidx_mx(cell,ncell)];
  const sunrealtype m2=y[pidx_my(cell,ncell)];
  const sunrealtype m3=y[pidx_mz(cell,ncell)];

  const int xl=pwrap_x(gx-1,ng), xr=pwrap_x(gx+1,ng);
  const int yu=pwrap_y(gy-1,ny), ydn=pwrap_y(gy+1,ny);
  const int lc=gy*ng+xl, rc=gy*ng+xr, uc=yu*ng+gx, dc=ydn*ng+gx;

  /*
   * Effective field at current y.
   * c_chb=0: no DMI contribution.
   * c_msk={1,0,0}: anisotropy only in H1.
   *
   * H1 = c_che*(m1_L+m1_R+m1_U+m1_D) + c_chk*m1
   * H2 = c_che*(m2_L+m2_R+m2_U+m2_D)
   * H3 = c_che*(m3_L+m3_R+m3_U+m3_D)
   */
  const sunrealtype h1 =
      pc_che*(y[pidx_mx(lc,ncell)]+y[pidx_mx(rc,ncell)]+
              y[pidx_mx(uc,ncell)]+y[pidx_mx(dc,ncell)]) +
      pc_chk*m1;

  const sunrealtype h2 =
      pc_che*(y[pidx_my(lc,ncell)]+y[pidx_my(rc,ncell)]+
              y[pidx_my(uc,ncell)]+y[pidx_my(dc,ncell)]);

  const sunrealtype h3 =
      pc_che*(y[pidx_mz(lc,ncell)]+y[pidx_mz(rc,ncell)]+
              y[pidx_mz(uc,ncell)]+y[pidx_mz(dc,ncell)]);

  const sunrealtype mh = m1*h1 + m2*h2 + m3*h3;

  /*
   * Self-coupling d_i = ∂(m·H)/∂m_i (neighbors frozen):
   *
   *   d1 = H1 + m1*(∂H1/∂m1) = H1 + m1*c_chk   ← x-axis: c_chk here
   *   d2 = H2                                    (no self term)
   *   d3 = H3                                    (no self term; z-axis had +c_chk*m3)
   */
  const sunrealtype d1 = h1 + pc_chk*m1;
  const sunrealtype d2 = h2;
  const sunrealtype d3 = h3;

  /*
   * 3×3 Jacobian block, row-major.
   * Row 0: ∂f1/∂(m1, m2, m3)
   * Row 1: ∂f2/∂(m1, m2, m3)
   * Row 2: ∂f3/∂(m1, m2, m3)
   *
   * Key differences from z-axis version:
   *   J[0][2]: was c_chg*(H2-m2)-c_alpha*d3*m1, now c_chg*H2-c_alpha*d3*m1
   *   J[1][2]: was c_chg*(m1-H1)-c_alpha*d3*m2, now -c_chg*H1-c_alpha*d3*m2
   *   J[2][2]: was c_alpha*(c_chk-d3*m3-mh),   now c_alpha*(-d3*m3-mh)
   *
   * These changes because ∂H3/∂m3 = 0 (no anisotropy in H3 for x-axis model).
   * The c_chk contribution now enters only through d1 in J[*][0] column.
   */
  const int b=cell*9;

  /* Row 0 */
  d_J[b+0] = pc_alpha*(-d1*m1 - mh);
  d_J[b+1] = -pc_chg*h3 - pc_alpha*d2*m1;
  d_J[b+2] =  pc_chg*h2 - pc_alpha*d3*m1;

  /* Row 1 */
  d_J[b+3] =  pc_chg*h3 - pc_alpha*d1*m2;
  d_J[b+4] = pc_alpha*(-d2*m2 - mh);
  d_J[b+5] = -pc_chg*h1 - pc_alpha*d3*m2;

  /* Row 2 */
  d_J[b+6] = -pc_chg*h2 - pc_alpha*d1*m3;
  d_J[b+7] =  pc_chg*h1 - pc_alpha*d2*m3;
  d_J[b+8] = pc_alpha*(-d3*m3 - mh);
}

/*
 * Kernel 2: build_Pinv_kernel — unchanged from original v2.
 * Reads d_J + gamma, computes P=(I-gamma*J)^{-1} via Cramer.
 * Called on EVERY psetup (jok=TRUE or FALSE) because gamma changes.
 */
__global__ static void build_Pinv_kernel(
    const sunrealtype* __restrict__ d_J,
    sunrealtype gamma,
    sunrealtype* __restrict__ d_Pinv,
    int ncell)
{
  const int cell=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(cell>=ncell) return;
  const int b=cell*9;

  const sunrealtype P00=1.0-gamma*d_J[b+0], P01=-gamma*d_J[b+1], P02=-gamma*d_J[b+2];
  const sunrealtype P10=-gamma*d_J[b+3],    P11=1.0-gamma*d_J[b+4], P12=-gamma*d_J[b+5];
  const sunrealtype P20=-gamma*d_J[b+6],    P21=-gamma*d_J[b+7],    P22=1.0-gamma*d_J[b+8];

  const sunrealtype det=P00*(P11*P22-P12*P21)-P01*(P10*P22-P12*P20)+P02*(P10*P21-P11*P20);
  const sunrealtype inv_det=(det!=0.0)?(1.0/det):1.0;

  d_Pinv[b+0]= inv_det*(P11*P22-P12*P21);
  d_Pinv[b+1]=-inv_det*(P01*P22-P02*P21);
  d_Pinv[b+2]= inv_det*(P01*P12-P02*P11);
  d_Pinv[b+3]=-inv_det*(P10*P22-P12*P20);
  d_Pinv[b+4]= inv_det*(P00*P22-P02*P20);
  d_Pinv[b+5]=-inv_det*(P00*P12-P02*P10);
  d_Pinv[b+6]= inv_det*(P10*P21-P11*P20);
  d_Pinv[b+7]=-inv_det*(P00*P21-P01*P20);
  d_Pinv[b+8]= inv_det*(P00*P11-P01*P10);
}

/* Kernel 3: psolve_kernel — unchanged */
__global__ static void psolve_kernel(
    const sunrealtype* __restrict__ r,
    sunrealtype*       __restrict__ z,
    const sunrealtype* __restrict__ Pinv,
    int ncell)
{
  const int cell=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(cell>=ncell) return;
  const int i0=pidx_mx(cell,ncell), i1=pidx_my(cell,ncell), i2=pidx_mz(cell,ncell);
  const sunrealtype r0=r[i0], r1=r[i1], r2=r[i2];
  const int b=cell*9;
  z[i0]=Pinv[b+0]*r0+Pinv[b+1]*r1+Pinv[b+2]*r2;
  z[i1]=Pinv[b+3]*r0+Pinv[b+4]*r1+Pinv[b+5]*r2;
  z[i2]=Pinv[b+6]*r0+Pinv[b+7]*r1+Pinv[b+8]*r2;
}

/* Public API */
PrecondData* Precond_Create(int ng, int ny, int ncell)
{
  PrecondData* pd=(PrecondData*)malloc(sizeof(PrecondData));
  if(!pd){fprintf(stderr,"precond: malloc failed\n");return NULL;}
  pd->ng=ng; pd->ny=ny; pd->ncell=ncell; pd->last_gamma=0.0;
  pd->d_J=NULL; pd->d_Pinv=NULL;

  const size_t sz=(size_t)ncell*9*sizeof(sunrealtype);
  if(cudaMalloc((void**)&pd->d_J,sz)!=cudaSuccess||
     cudaMalloc((void**)&pd->d_Pinv,sz)!=cudaSuccess){
    fprintf(stderr,"precond: cudaMalloc failed\n");
    Precond_Destroy(pd); return NULL;
  }
  printf("[Precond] x-axis anisotropy, no DMI. Storage: %zu MB each.\n",sz/(1024*1024));
  return pd;
}

void Precond_Destroy(PrecondData* pd)
{
  if(!pd) return;
  if(pd->d_J)    cudaFree(pd->d_J);
  if(pd->d_Pinv) cudaFree(pd->d_Pinv);
  free(pd);
}

int PrecondSetup(sunrealtype t,
                 N_Vector y, N_Vector fy,
                 sunbooleantype jok, sunbooleantype* jcurPtr,
                 sunrealtype gamma, void* user_data)
{
  (void)t;(void)fy;
  PrecondData* pd=*(PrecondData**)user_data;

  if(!jok){
    /* Rebuild J from current y (expensive: reads stencil) */
    const sunrealtype* yd=N_VGetDeviceArrayPointer_Cuda(y);
    const dim3 block2d(16,8);
    const dim3 grid2d((pd->ng+15)/16,(pd->ny+7)/8);
    build_J_kernel<<<grid2d,block2d>>>(yd,pd->d_J,pd->ng,pd->ny,pd->ncell);
    if(cudaPeekAtLastError()!=cudaSuccess){
      fprintf(stderr,"precond: build_J_kernel failed\n"); return -1;}
    *jcurPtr=SUNTRUE;
  } else {
    *jcurPtr=SUNFALSE;
  }

  /* ALWAYS rebuild P^{-1} — gamma changes every Newton step */
  const int block1d=256, grid1d=(pd->ncell+255)/256;
  build_Pinv_kernel<<<grid1d,block1d>>>(pd->d_J,gamma,pd->d_Pinv,pd->ncell);
  if(cudaPeekAtLastError()!=cudaSuccess){
    fprintf(stderr,"precond: build_Pinv_kernel failed\n"); return -1;}

  pd->last_gamma=gamma;
  return 0;
}

int PrecondSolve(sunrealtype t,
                 N_Vector y, N_Vector fy,
                 N_Vector r, N_Vector z,
                 sunrealtype gamma, sunrealtype delta,
                 int lr, void* user_data)
{
  (void)t;(void)y;(void)fy;(void)gamma;(void)delta;(void)lr;
  PrecondData* pd=*(PrecondData**)user_data;
  const sunrealtype* rd=N_VGetDeviceArrayPointer_Cuda(r);
  sunrealtype*       zd=N_VGetDeviceArrayPointer_Cuda(z);
  const int block=256, grid=(pd->ncell+255)/256;
  psolve_kernel<<<grid,block>>>(rd,zd,pd->d_Pinv,pd->ncell);
  if(cudaPeekAtLastError()!=cudaSuccess){
    fprintf(stderr,"precond: psolve_kernel failed\n"); return -1;}
  return 0;
}
