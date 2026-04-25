/**
 * 2D LLG solver — Voronoi polycrystal + dead-grain holes + FFT demag
 * Drop-in replacement for 2d_i3/2d_fft.cu
 *
 * ─── Geometry ────────────────────────────────────────────────────────
 *   - NUM_GRAINS Voronoi seeds (stratified jitter, periodic)
 *   - DEAD_GRAIN_FRAC of grains are killed → polygonal holes
 *   - soft tanh boundary only at hole boundaries
 *   - grain↔grain interfaces stay sharp (m discontinuity = mesh look)
 *
 * ─── Physics ─────────────────────────────────────────────────────────
 *   - per-cell easy axis msk(x,y) (uniform inside a grain, random per grain)
 *   - proper uniaxial anisotropy:  h_α += msk_α · (chk · (m·msk) + cha)
 *   - exchange / DMI use weighted neighbors w·n̂  (via y_eff buffer)
 *   - demag: FFT( w·n̂ ) — windowed magnetization, no leakage into holes
 *   - yd = LLG(m, h_eff) · w[self]   (deep-hole frozen, soft boundary)
 *
 * ─── PMPP-clean GPU layout ───────────────────────────────────────────
 *   - DENSE launch, no `if (active)` in hot path
 *   - msk stored in SoA layout identical to y → reads are coalesced
 *   - one fused kernel: exchange + per-cell aniso + DMI + demag + LLG + w
 *   - apply_weight_kernel pre-pass writes y_eff once, reused by demag and RHS
 *
 * Build: make run-demag PRINT=1
 * Plot : same mdyn2D.m on output.txt (m·w → arrows fade in holes)
 */

#include <cvode/cvode.h>
#include <cuda_runtime.h>
#include <math.h>
#include <nvector/nvector_cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>
#include <sundials/sundials_iterative.h>

#include "deferred_nvector.h"
#include "precond.h"
#include "jtv.h"
#include "demag_fft.h"

#define GROUPSIZE 3

/* ─── Solver tuning ──────────────────────────────────────────────── */
#ifndef KRYLOV_DIM
#define KRYLOV_DIM 0
#endif
#ifndef MAX_BDF_ORDER
#define MAX_BDF_ORDER 5
#endif
#ifndef RTOL_VAL
#define RTOL_VAL 1.0e-5
#endif
#ifndef ATOL_VAL
#define ATOL_VAL 1.0e-5
#endif

/* ─── Demag controls ─────────────────────────────────────────────── */
#ifndef DEMAG_STRENGTH
#define DEMAG_STRENGTH 0.0
#endif
#ifndef DEMAG_THICK
#define DEMAG_THICK 1.0
#endif

#define RTOL  SUN_RCONST(RTOL_VAL)
#define ATOL1 SUN_RCONST(ATOL_VAL)
#define ATOL2 SUN_RCONST(ATOL_VAL)
#define ATOL3 SUN_RCONST(ATOL_VAL)
#define T0   SUN_RCONST(0.0)
#define T1   SUN_RCONST(0.1)
#define ZERO SUN_RCONST(0.0)

#ifndef T_TOTAL
#define T_TOTAL 1000.0
#endif
#ifndef ENABLE_OUTPUT
#define ENABLE_OUTPUT 0
#endif
#ifndef BLOCK_X
#define BLOCK_X 16
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 8
#endif

/* ─── Initial-condition knobs ────────────────────────────────────── */
#ifndef INIT_RANDOM_EPS
#define INIT_RANDOM_EPS 0.05
#endif
#ifndef INIT_RANDOM_SEED
#define INIT_RANDOM_SEED 12345
#endif
#ifndef EARLY_SAVE_UNTIL
#define EARLY_SAVE_UNTIL 80.0
#endif
#ifndef EARLY_SAVE_EVERY
#define EARLY_SAVE_EVERY 5
#endif
#ifndef LATE_SAVE_EVERY
#define LATE_SAVE_EVERY 100
#endif

/* ─── Polycrystal knobs ──────────────────────────────────────────── */
#ifndef NUM_GRAINS
#define NUM_GRAINS 72                /* was 48 — finer mesh */
#endif
#ifndef DEAD_GRAIN_FRAC
#define DEAD_GRAIN_FRAC 0.16         /* was 0.18 */
#endif
#ifndef HOLE_SEED
#define HOLE_SEED 20251104
#endif
#ifndef MASK_EPS_CELLS
#define MASK_EPS_CELLS 2.2           /* was 1.5 — softer boundaries */
#endif
#ifndef GRAIN_Z_BIAS
#define GRAIN_Z_BIAS 1.6             /* >1 = bias easy-axes toward ±z */
#endif
#ifndef IC_CORE_MZ
#define IC_CORE_MZ 0.95              /* grain-core out-of-plane amplitude */
#endif

#ifndef GRID_NX
#define GRID_NX 768
#endif
#ifndef GRID_NY
#define GRID_NY 256
#endif

/* ─── Material constants (device constant memory) ────────────────── */
/* c_msk_default kept for reference; the kernel actually reads per-cell d_msk.
 * c_nsk (DMI direction) stays uniform along x. */
__constant__ sunrealtype c_nsk[3] = {SUN_RCONST(1.0), SUN_RCONST(0.0), SUN_RCONST(0.0)};
__constant__ sunrealtype c_chk   = SUN_RCONST(1.0);   /* was 4.0 */
__constant__ sunrealtype c_che   = SUN_RCONST(4.0);
__constant__ sunrealtype c_alpha = SUN_RCONST(0.2);
__constant__ sunrealtype c_chg   = SUN_RCONST(1.0);
__constant__ sunrealtype c_cha   = SUN_RCONST(0.0);
__constant__ sunrealtype c_chb   = SUN_RCONST(0.6);   /* was 0.3 */
/* c_nsk removed — DMI direction is now per-cell, see d_nsk */

/* ─── Error checking ─────────────────────────────────────────────── */
#define CHECK_CUDA(call)                                                     \
  do { cudaError_t _err = (call);                                            \
       if (_err != cudaSuccess) {                                            \
         fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                 cudaGetErrorString(_err)); exit(EXIT_FAILURE); } } while (0)

#define CHECK_SUNDIALS(call)                                                 \
  do { int _flag = (call);                                                   \
       if (_flag < 0) {                                                      \
         fprintf(stderr, "SUNDIALS error at %s:%d: flag = %d\n",             \
                 __FILE__, __LINE__, _flag); exit(EXIT_FAILURE); } } while (0)

/* ─── UserData ───────────────────────────────────────────────────── */
/* First 72 bytes (pd, demag, d_hdmag, ints, doubles) MUST match the
 * mirrors in precond.cu / jtv.cu byte-for-byte.  All extension fields
 * sit AFTER nzz0 and are invisible to those casts. */
typedef struct {
  PrecondData  *pd;          /* 0  */
  DemagData    *demag;       /* 8  */
  sunrealtype  *d_hdmag;     /* 16 */
  int nx;                    /* 24 */
  int ny;                    /* 28 */
  int ng;                    /* 32 */
  int ncell;                 /* 36 */
  int neq;                   /* 40 */
  /* 4-byte pad */
  double nxx0;               /* 48 */
  double nyy0;               /* 56 */
  double nzz0;               /* 64 */
  /* === extensions (invisible to precond/jtv mirrors) === */
  sunrealtype  *d_w;         /* per-cell soft weight */
  sunrealtype  *d_y_eff;     /* scratch w·y, 3*ncell */
  sunrealtype  *d_msk;       /* per-cell easy axis, SoA 3*ncell */
  sunrealtype  *d_nsk;       /* per-cell DMI direction */
  double       *h_w;
  double       *h_msk;       /* host SoA copy */
  double       *h_nsk;       /* host SoA copy */
  int          *h_grain_id;  /* host: which grain each cell belongs to */
  int           num_dead;
} UserData;

/* ─── SoA indexing helpers ───────────────────────────────────────── */
__host__ __device__ static inline int idx_mx(int cell, int ncell) { return cell; }
__host__ __device__ static inline int idx_my(int cell, int ncell) { return ncell + cell; }
__host__ __device__ static inline int idx_mz(int cell, int ncell) { return 2*ncell + cell; }
__host__ __device__ static inline int wrap_x(int x, int ng) {
  return (x < 0) ? (x + ng) : ((x >= ng) ? (x - ng) : x);
}
__host__ __device__ static inline int wrap_y(int y, int ny) {
  return (y < 0) ? (y + ny) : ((y >= ny) ? (y - ny) : y);
}

/* ─── Polycrystal: host-side generation ──────────────────────────── */
typedef struct {
  double sx, sy;          /* seed center */
  double ax, ay, az;      /* easy axis (unit) */
  double nx_, ny_, nz_;   /* DMI direction (unit, in-plane preferred) */
  int    dead;
} GrainSpec;

static GrainSpec g_grains[NUM_GRAINS];
static double rand_unit(void) { return (double)rand() / (double)RAND_MAX; }

static void generate_grains(int ng, int ny) {
  srand((unsigned)HOLE_SEED);

  /* stratified-jitter seed placement */
  int gx = (int)ceil(sqrt((double)NUM_GRAINS * (double)ng / (double)ny));
  if (gx < 1) gx = 1;
  int gy = (NUM_GRAINS + gx - 1) / gx;
  const double dxs = (double)ng / (double)gx;
  const double dys = (double)ny / (double)gy;

  int idx = 0;
  for (int j = 0; j < gy && idx < NUM_GRAINS; j++)
    for (int i = 0; i < gx && idx < NUM_GRAINS; i++) {
      g_grains[idx].sx = ((double)i + 0.15 + 0.7 * rand_unit()) * dxs;
      g_grains[idx].sy = ((double)j + 0.15 + 0.7 * rand_unit()) * dys;
      while (g_grains[idx].sx <  0.0)        g_grains[idx].sx += (double)ng;
      while (g_grains[idx].sx >= (double)ng) g_grains[idx].sx -= (double)ng;
      while (g_grains[idx].sy <  0.0)        g_grains[idx].sy += (double)ny;
      while (g_grains[idx].sy >= (double)ny) g_grains[idx].sy -= (double)ny;
      idx++;
    }
  while (idx < NUM_GRAINS) {
    g_grains[idx].sx = rand_unit() * (double)ng;
    g_grains[idx].sy = rand_unit() * (double)ny;
    idx++;
  }

  for (int g = 0; g < NUM_GRAINS; g++) {
    /* easy axis: power-bias toward ±z so mz patches stand out
     *   z = sign · |u|^(1/bias),    bias>1 pushes |z|→1 */
    const double u = rand_unit();
    const double sign = (rand_unit() < 0.5) ? -1.0 : 1.0;
    const double z = sign * pow(u, 1.0 / (double)GRAIN_Z_BIAS);
    const double phi = 2.0 * M_PI * rand_unit();
    const double rxy = sqrt(fmax(0.0, 1.0 - z*z));
    g_grains[g].ax = rxy * cos(phi);
    g_grains[g].ay = rxy * sin(phi);
    g_grains[g].az = z;

    /* DMI direction: random unit in xy-plane (Néel-like wall direction) */
    const double psi = 2.0 * M_PI * rand_unit();
    g_grains[g].nx_ = cos(psi);
    g_grains[g].ny_ = sin(psi);
    g_grains[g].nz_ = 0.0;

    g_grains[g].dead = (rand_unit() < (double)DEAD_GRAIN_FRAC) ? 1 : 0;
  }
}

/* Nearest grain (with periodic 3×3 replication). */
static int nearest_grain(double x, double y, int ng, int ny) {
  int best = 0;
  double best_d2 = 1.0e30;
  for (int g = 0; g < NUM_GRAINS; g++) {
    for (int dyp = -1; dyp <= 1; dyp++) {
      for (int dxp = -1; dxp <= 1; dxp++) {
        const double sx = g_grains[g].sx + (double)dxp * (double)ng;
        const double sy = g_grains[g].sy + (double)dyp * (double)ny;
        const double dx = x - sx, dy = y - sy;
        const double d2 = dx*dx + dy*dy;
        if (d2 < best_d2) { best_d2 = d2; best = g; }
      }
    }
  }
  return best;
}

/* Build grain_id, soft weight w, and per-cell easy-axis msk in one go. */
static void build_polycrystal(int ng, int ny, UserData *udata) {
  const int ncell = ng * ny;
  generate_grains(ng, ny);

  /* Pass 1: assign grain to every cell. */
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < ng; i++) {
      udata->h_grain_id[j*ng + i] =
        nearest_grain((double)i + 0.5, (double)j + 0.5, ng, ny);
    }
  }

  /* Pass 2: distance to nearest dead-grain cell (PBC, brute-force). */
  int n_dead_grains = 0;
  for (int g = 0; g < NUM_GRAINS; g++) if (g_grains[g].dead) n_dead_grains++;
  udata->num_dead = n_dead_grains;

  int n_hole_cells = 0;
  for (int k = 0; k < ncell; k++)
    if (g_grains[udata->h_grain_id[k]].dead) n_hole_cells++;

  if (n_hole_cells == 0) {
    for (int k = 0; k < ncell; k++) udata->h_w[k] = 1.0;
  } else {
    int *hx = (int*)malloc((size_t)n_hole_cells * sizeof(int));
    int *hy = (int*)malloc((size_t)n_hole_cells * sizeof(int));
    if (!hx || !hy) { fprintf(stderr, "hx/hy alloc failed\n"); exit(1); }
    int idx = 0;
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < ng; i++)
        if (g_grains[udata->h_grain_id[j*ng+i]].dead)
          { hx[idx]=i; hy[idx]=j; idx++; }

    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < ng; i++) {
        const int cell = j * ng + i;
        if (g_grains[udata->h_grain_id[cell]].dead) {
          udata->h_w[cell] = 0.0;
          continue;
        }
        double d2_min = 1.0e30;
        for (int k = 0; k < n_hole_cells; k++) {
          int ddx = i - hx[k];
          int ddy = j - hy[k];
          if (ddx >  ng/2) ddx -= ng;
          if (ddx < -ng/2) ddx += ng;
          if (ddy >  ny/2) ddy -= ny;
          if (ddy < -ny/2) ddy += ny;
          const double d2 = (double)(ddx*ddx + ddy*ddy);
          if (d2 < d2_min) d2_min = d2;
        }
        const double d = sqrt(d2_min) - 0.5;   /* sub-cell offset */
        double w = 0.5 * (1.0 + tanh(d / (double)MASK_EPS_CELLS));
        if (w < 1.0e-3)        w = 0.0;
        if (w > 1.0 - 1.0e-3)  w = 1.0;
        udata->h_w[cell] = w;
      }
    }
    free(hx); free(hy);
  }

  /* Pass 3: per-cell easy axis (SoA — same layout as y). */
  for (int k = 0; k < ncell; k++) {
    const int g = udata->h_grain_id[k];
    udata->h_msk[idx_mx(k, ncell)] = g_grains[g].ax;
    udata->h_msk[idx_my(k, ncell)] = g_grains[g].ay;
    udata->h_msk[idx_mz(k, ncell)] = g_grains[g].az;
  }

  /* Pass 4: per-cell DMI direction (SoA) */
  for (int k = 0; k < ncell; k++) {
    const int g = udata->h_grain_id[k];
    udata->h_nsk[idx_mx(k, ncell)] = g_grains[g].nx_;
    udata->h_nsk[idx_my(k, ncell)] = g_grains[g].ny_;
    udata->h_nsk[idx_mz(k, ncell)] = g_grains[g].nz_;
  }
}

/* ─── Output helpers ─────────────────────────────────────────────── */
static void write_aux_files(int ng, int ny,
                            const double *h_w, const int *h_grain_id) {
  FILE *fm = fopen("mask.txt", "w");
  if (fm) {
    fprintf(fm, "%d %d\n", ng, ny);
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < ng; i++)
        fprintf(fm, "%f\n", h_w[j*ng + i]);
    fclose(fm);
  }
  FILE *fg = fopen("grain_id.txt", "w");
  if (fg) {
    fprintf(fg, "%d %d\n", ng, ny);
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < ng; i++)
        fprintf(fg, "%d\n", h_grain_id[j*ng + i]);
    fclose(fg);
  }
}

/* ─── y_eff = w · y  (one pass, coalesced) ───────────────────────── */
__global__ static void apply_weight_kernel(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ w,
    sunrealtype*       __restrict__ y_eff,
    int ncell)
{
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= ncell) return;
  const sunrealtype ww = w[cell];
  y_eff[cell]             = ww * y[cell];
  y_eff[ncell + cell]     = ww * y[ncell + cell];
  y_eff[2 * ncell + cell] = ww * y[2 * ncell + cell];
}

/* ─── |m|=1 normalization on cells outside deep holes ───────────── */
__global__ static void normalize_m_kernel_w(
    sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ w,
    int ncell)
{
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= ncell) return;
  if (w[cell] < SUN_RCONST(1.0e-3)) return;

  const int mx = idx_mx(cell, ncell);
  const int my = idx_my(cell, ncell);
  const int mz = idx_mz(cell, ncell);
  const sunrealtype m1 = y[mx], m2 = y[my], m3 = y[mz];
  const sunrealtype ymp = sqrt(m1*m1 + m2*m2 + m3*m3);
  if (ymp > SUN_RCONST(1.0e-30)) {
    const sunrealtype inv = SUN_RCONST(1.0) / ymp;
    y[mx] = m1 * inv; y[my] = m2 * inv; y[mz] = m3 * inv;
  }
}

/* ─── Unified RHS: per-cell easy axis + weighted neighbors ──────── */
__global__ static void f_kernel_unified_polycrystal(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ y_eff,
    const sunrealtype* __restrict__ w,
    const sunrealtype* __restrict__ msk,     /* SoA */
    const sunrealtype* __restrict__ nsk,     /* SoA, per-cell DMI dir */
    const sunrealtype* __restrict__ h_dmag,
    sunrealtype*       __restrict__ yd,
    int ng, int ny, int ncell)
{
  const int gx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gy = blockIdx.y * blockDim.y + threadIdx.y;
  if (gx >= ng || gy >= ny) return;

  const int cell = gy * ng + gx;
  const int mx_i = idx_mx(cell, ncell);
  const int my_i = idx_my(cell, ncell);
  const int mz_i = idx_mz(cell, ncell);

  const int xl  = wrap_x(gx - 1, ng);
  const int xr  = wrap_x(gx + 1, ng);
  const int yu  = wrap_y(gy - 1, ny);
  const int ydn = wrap_y(gy + 1, ny);
  const int lc = gy  * ng + xl;
  const int rc = gy  * ng + xr;
  const int uc = yu  * ng + gx;
  const int dc = ydn * ng + gx;

  const sunrealtype m1 = y[mx_i], m2 = y[my_i], m3 = y[mz_i];
  const sunrealtype w_self = w[cell];

  const sunrealtype mskx = msk[mx_i], msky = msk[my_i], mskz = msk[mz_i];
  const sunrealtype nskx = nsk[mx_i], nsky = nsk[my_i], nskz = nsk[mz_i];

  const sunrealtype lx1 = y_eff[idx_mx(lc, ncell)];
  const sunrealtype lx2 = y_eff[idx_my(lc, ncell)];
  const sunrealtype lx3 = y_eff[idx_mz(lc, ncell)];
  const sunrealtype rx1 = y_eff[idx_mx(rc, ncell)];
  const sunrealtype rx2 = y_eff[idx_my(rc, ncell)];
  const sunrealtype rx3 = y_eff[idx_mz(rc, ncell)];
  const sunrealtype ux1 = y_eff[idx_mx(uc, ncell)];
  const sunrealtype ux2 = y_eff[idx_my(uc, ncell)];
  const sunrealtype ux3 = y_eff[idx_mz(uc, ncell)];
  const sunrealtype dx1 = y_eff[idx_mx(dc, ncell)];
  const sunrealtype dx2 = y_eff[idx_my(dc, ncell)];
  const sunrealtype dx3 = y_eff[idx_mz(dc, ncell)];

  /* DMI uses x-neighbors projected on per-cell DMI direction */
  const sunrealtype dmi_x = lx1 + rx1;
  const sunrealtype dmi_y = lx2 + rx2;
  const sunrealtype dmi_z = lx3 + rx3;

  const sunrealtype mdotmsk      = m1*mskx + m2*msky + m3*mskz;
  const sunrealtype aniso_factor = c_chk * mdotmsk + c_cha;

  const sunrealtype h1 =
      c_che * (lx1 + rx1 + ux1 + dx1)
    + mskx * aniso_factor
    + c_chb * nskx * dmi_x
    + h_dmag[mx_i];
  const sunrealtype h2 =
      c_che * (lx2 + rx2 + ux2 + dx2)
    + msky * aniso_factor
    + c_chb * nsky * dmi_y
    + h_dmag[my_i];
  const sunrealtype h3 =
      c_che * (lx3 + rx3 + ux3 + dx3)
    + mskz * aniso_factor
    + c_chb * nskz * dmi_z
    + h_dmag[mz_i];

  const sunrealtype mh = m1*h1 + m2*h2 + m3*h3;
  yd[mx_i] = w_self * (c_chg * (m3*h2 - m2*h3) + c_alpha * (h1 - mh*m1));
  yd[my_i] = w_self * (c_chg * (m1*h3 - m3*h1) + c_alpha * (h2 - mh*m2));
  yd[mz_i] = w_self * (c_chg * (m2*h1 - m1*h2) + c_alpha * (h3 - mh*m3));
}

/* ─── RHS wrapper ────────────────────────────────────────────────── */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
  (void)t;
  UserData* udata = (UserData*)user_data;
  sunrealtype* ydata    = N_VGetDeviceArrayPointer_Cuda(y);
  sunrealtype* ydotdata = N_VGetDeviceArrayPointer_Cuda(ydot);

  if (BLOCK_X * BLOCK_Y > 1024) {
    fprintf(stderr, "Invalid block size: %d > 1024\n", BLOCK_X * BLOCK_Y);
    return -1;
  }

  /* 0) |n̂|=1 outside deep holes */
  {
    const int nb = 256;
    const int gd = (udata->ncell + nb - 1) / nb;
    normalize_m_kernel_w<<<gd, nb>>>(ydata, udata->d_w, udata->ncell);
  }
  /* 1) y_eff = w · y */
  {
    const int nb = 256;
    const int gd = (udata->ncell + nb - 1) / nb;
    apply_weight_kernel<<<gd, nb>>>(ydata, udata->d_w, udata->d_y_eff,
                                    udata->ncell);
  }
  /* 2) demag on windowed magnetization */
  if (udata->demag && DEMAG_STRENGTH > 0.0) {
    Demag_Apply(udata->demag, (const double*)udata->d_y_eff,
                              (double*)udata->d_hdmag);
  } else {
    cudaMemsetAsync(udata->d_hdmag, 0,
                    (size_t)3 * udata->ncell * sizeof(sunrealtype), 0);
  }
  /* 3) unified RHS */
  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid((udata->ng + block.x - 1)/block.x,
            (udata->ny + block.y - 1)/block.y);
  f_kernel_unified_polycrystal<<<grid, block>>>(
      ydata, udata->d_y_eff, udata->d_w,
      udata->d_msk, udata->d_nsk,
      udata->d_hdmag, ydotdata,
      udata->ng, udata->ny, udata->ncell);

  cudaError_t cuerr = cudaPeekAtLastError();
  if (cuerr != cudaSuccess) {
    fprintf(stderr, ">>> ERROR in f: %s\n", cudaGetErrorString(cuerr));
    return -1;
  }
  return 0;
}

/* ─── Final-stats reporter ───────────────────────────────────────── */
static void PrintFinalStats(void* cvode_mem, SUNLinearSolver LS) {
  (void)LS;
  long int nst, nfe, nsetups, nni, ncfn, netf, nge, nli, nlcf, njvevals;
  CVodeGetNumSteps(cvode_mem, &nst);
  CVodeGetNumRhsEvals(cvode_mem, &nfe);
  CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
  CVodeGetNumErrTestFails(cvode_mem, &netf);
  CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
  CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
  CVodeGetNumGEvals(cvode_mem, &nge);
  CVodeGetNumLinIters(cvode_mem, &nli);
  CVodeGetNumLinConvFails(cvode_mem, &nlcf);
  CVodeGetNumJtimesEvals(cvode_mem, &njvevals);
  printf("\nFinal Statistics:\n");
  printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld ", nst, nfe, nsetups);
  printf("nni = %-6ld ncfn = %-6ld netf = %-6ld nge = %ld\n",
         nni, ncfn, netf, nge);
  printf("nli = %-6ld nlcf = %-6ld njvevals = %ld\n", nli, nlcf, njvevals);
}

#if ENABLE_OUTPUT
/* WriteFrame outputs m·w → arrows fade smoothly in holes (no grid/ghost). */
static void WriteFrame(FILE* fp, sunrealtype t,
                       int nx, int ny, int ng, int ncell,
                       N_Vector y, const double *h_w)
{
  N_VCopyFromDevice_Cuda(y);
  sunrealtype* ydata = N_VGetHostArrayPointer_Cuda(y);
  fprintf(fp, "%f %d %d\n", (double)t, nx, ny);
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < ng; i++) {
      const int c = j * ng + i;
      const double w = h_w[c];
      fprintf(fp, "%f %f %f\n",
              w * (double)ydata[idx_mx(c, ncell)],
              w * (double)ydata[idx_my(c, ncell)],
              w * (double)ydata[idx_mz(c, ncell)]);
    }
  }
  fprintf(fp, "\n");
}
static int ShouldWriteFrame(long int iout, sunrealtype t) {
  if (t <= SUN_RCONST(EARLY_SAVE_UNTIL)) return (iout % EARLY_SAVE_EVERY) == 0;
  return (iout % LATE_SAVE_EVERY) == 0;
}
#endif

/* ─── main ───────────────────────────────────────────────────────── */
int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

  SUNContext sunctx = NULL;
  sunrealtype *ydata = NULL, *abstol_data = NULL;
  sunrealtype t = T0, tout = T1;
  sunrealtype ttotal = SUN_RCONST(T_TOTAL);

  N_Vector y = NULL, abstol = NULL;
  SUNLinearSolver LS = NULL;
  SUNNonlinearSolver NLS = NULL;
  void* cvode_mem = NULL;

  int retval;
  long int iout, NOUT;
  UserData udata;
  memset(&udata, 0, sizeof(udata));

  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;

  const int nx = GRID_NX;
  const int ny = GRID_NY;
  if (nx % GROUPSIZE != 0) {
    fprintf(stderr, "nx must be a multiple of GROUPSIZE=%d\n", GROUPSIZE);
    return 1;
  }
  const int ng    = nx / GROUPSIZE;
  const int ncell = ng * ny;
  const int neq   = 3 * ncell;

#if ENABLE_OUTPUT
  FILE* fp = fopen("output.txt", "w");
  if (!fp) { fprintf(stderr, "Error opening output file.\n"); return 1; }
  setvbuf(fp, NULL, _IOFBF, 1 << 20);
#else
  FILE* fp = NULL; (void)fp;
#endif

  udata.nx = nx; udata.ny = ny; udata.ng = ng;
  udata.ncell = ncell; udata.neq = neq;

  CHECK_SUNDIALS(SUNContext_Create(SUN_COMM_NULL, &sunctx));

  /* ── Polycrystal: grain ids, soft weight, per-cell easy axis ──── */
  udata.h_w        = (double*)malloc((size_t)ncell * sizeof(double));
  udata.h_msk      = (double*)malloc((size_t)3 * ncell * sizeof(double));
  udata.h_nsk      = (double*)malloc((size_t)3 * ncell * sizeof(double));
  udata.h_grain_id = (int*)   malloc((size_t)ncell * sizeof(int));
  if (!udata.h_w || !udata.h_msk || !udata.h_nsk || !udata.h_grain_id) {
    fprintf(stderr, "host alloc failed\n"); return 1;
  }
  build_polycrystal(ng, ny, &udata);
  write_aux_files(ng, ny, udata.h_w, udata.h_grain_id);

  {
    double w_sum = 0.0;
    int n_full = 0, n_hole = 0, n_boundary = 0;
    for (int k = 0; k < ncell; k++) {
      const double w = udata.h_w[k];
      w_sum += w;
      if (w <= 1.0e-3)            n_hole++;
      else if (w >= 1.0 - 1.0e-3) n_full++;
      else                         n_boundary++;
    }
    printf("[Polycrystal] %d grains  (%d dead → polygonal holes), seed=%d\n",
           NUM_GRAINS, udata.num_dead, (int)HOLE_SEED);
    printf("[Polycrystal] cells: %d total, %d full, %d soft-boundary, %d deep-hole\n",
           ncell, n_full, n_boundary, n_hole);
    printf("[Polycrystal] effective active fraction (sum w / N) = %.4f\n",
           w_sum / (double)ncell);
    printf("[Polycrystal] eps=%.2f cells, output: mask.txt grain_id.txt\n",
           (double)MASK_EPS_CELLS);
  }

  /* upload weight + msk + alloc y_eff scratch */
  /* upload weight + msk + nsk + alloc y_eff scratch */
  CHECK_CUDA(cudaMalloc((void**)&udata.d_w,
                        (size_t)ncell * sizeof(sunrealtype)));
  CHECK_CUDA(cudaMemcpy(udata.d_w, udata.h_w,
                        (size_t)ncell * sizeof(sunrealtype),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc((void**)&udata.d_msk,
                        (size_t)3 * ncell * sizeof(sunrealtype)));
  CHECK_CUDA(cudaMemcpy(udata.d_msk, udata.h_msk,
                        (size_t)3 * ncell * sizeof(sunrealtype),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc((void**)&udata.d_nsk,
                        (size_t)3 * ncell * sizeof(sunrealtype)));
  CHECK_CUDA(cudaMemcpy(udata.d_nsk, udata.h_nsk,
                        (size_t)3 * ncell * sizeof(sunrealtype),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc((void**)&udata.d_y_eff,
                        (size_t)3 * ncell * sizeof(sunrealtype)));
  CHECK_CUDA(cudaMemset(udata.d_y_eff, 0,
                        (size_t)3 * ncell * sizeof(sunrealtype)));

  /* preconditioner — same as before; precond's anisotropy assumption is
   * approximate for varying easy axis but GMRES tolerates it. */
  udata.pd = Precond_Create(ng, ny, ncell);
  if (!udata.pd) { fprintf(stderr, "Precond_Create failed\n"); return 1; }

  CHECK_CUDA(cudaMalloc((void**)&udata.d_hdmag,
                        (size_t)3 * ncell * sizeof(sunrealtype)));
  CHECK_CUDA(cudaMemset(udata.d_hdmag, 0,
                        (size_t)3 * ncell * sizeof(sunrealtype)));

  const double dstr = (double)DEMAG_STRENGTH;
  const double dthk = (double)DEMAG_THICK;
  if (dstr > 0.0) {
    udata.demag = Demag_Init(ng, ny, dthk, dstr);
    if (!udata.demag) {
      fprintf(stderr, "Demag_Init failed\n");
      Precond_Destroy(udata.pd);
      cudaFree(udata.d_hdmag);
      cudaFree(udata.d_w); cudaFree(udata.d_y_eff);
      cudaFree(udata.d_msk); cudaFree(udata.d_nsk);
      return 1;
    }
    Demag_GetSelfCoupling(udata.demag,
                          &udata.nxx0, &udata.nyy0, &udata.nzz0);
    printf("[main] Demag self-coupling (scaled): "
           "nxx0=%.4e nyy0=%.4e nzz0=%.4e\n",
           udata.nxx0, udata.nyy0, udata.nzz0);
  }

  y      = N_VNew_Cuda(neq, sunctx);
  abstol = N_VNew_Cuda(neq, sunctx);
  if (!y || !abstol) { fprintf(stderr, "N_Vector alloc failed\n"); goto cleanup; }
  FusedNVec_Init(y);

  ydata       = N_VGetHostArrayPointer_Cuda(y);
  abstol_data = N_VGetHostArrayPointer_Cuda(abstol);

  /* IC: grain core has out-of-plane mz; smoothly tilt to in-plane near
   * grain boundaries.  This produces a polycrystal "bumpy" mz landscape
   * before LLG even starts.  Cells inside dead grains stay zero. */
  {
    /* per-grain effective radius — distance from cell to its seed */
    srand((unsigned)INIT_RANDOM_SEED);
    const double core = (double)IC_CORE_MZ;
    const double eps_n = 0.05;

    /* compute, for every cell, distance to its grain seed (PBC) */
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < ng; i++) {
        const int k = j * ng + i;
        const int g = udata.h_grain_id[k];
        const int mx_i = idx_mx(k, ncell);
        const int my_i = idx_my(k, ncell);
        const int mz_i = idx_mz(k, ncell);

        if (udata.h_w[k] <= 1.0e-3) {
          ydata[mx_i] = ZERO; ydata[my_i] = ZERO; ydata[mz_i] = ZERO;
          abstol_data[mx_i] = ATOL1;
          abstol_data[my_i] = ATOL2;
          abstol_data[mz_i] = ATOL3;
          continue;
        }

        /* signed-PBC distance to seed */
        double ddx = (double)i + 0.5 - g_grains[g].sx;
        double ddy = (double)j + 0.5 - g_grains[g].sy;
        if (ddx >  ng/2.0) ddx -= ng;
        if (ddx < -ng/2.0) ddx += ng;
        if (ddy >  ny/2.0) ddy -= ny;
        if (ddy < -ny/2.0) ddy += ny;
        const double r = sqrt(ddx*ddx + ddy*ddy);

        /* normalized radius (saturate at ~1 grain radius ≈ 12 cells) */
        double s = r / 14.0;
        if (s > 1.0) s = 1.0;

        /* mz: full at core (sign of grain easy-axis z), → 0 at edge */
        double sign_z = (g_grains[g].az >= 0.0) ? 1.0 : -1.0;
        if (fabs(g_grains[g].az) < 0.05) sign_z = 0.0;  /* in-plane grain */

        double mz0 = sign_z * core * (1.0 - s*s);
        /* in-plane part: tangent to "vortex" + grain easy-axis xy bias */
        double mperp = sqrt(fmax(0.0, 1.0 - mz0*mz0));
        double tx = -ddy, ty = ddx;
        const double tn = sqrt(tx*tx + ty*ty);
        if (tn > 1.0e-12) { tx /= tn; ty /= tn; }
        else { tx = 1.0; ty = 0.0; }
        /* blend tangent with grain easy axis xy */
        const double bx = g_grains[g].ax;
        const double by = g_grains[g].ay;
        const double bn = sqrt(bx*bx + by*by);
        double ex, ey;
        if (bn > 1.0e-6) { ex = bx/bn; ey = by/bn; }
        else             { ex = 1.0;   ey = 0.0;   }
        const double blend = s;     /* core = vortex, edge = easy axis */
        double pxx = (1.0-blend)*tx + blend*ex;
        double pyy = (1.0-blend)*ty + blend*ey;
        const double pn = sqrt(pxx*pxx + pyy*pyy);
        if (pn > 1.0e-12) { pxx /= pn; pyy /= pn; }

        double mx0 = mperp * pxx + eps_n * (2.0*rand_unit() - 1.0);
        double my0 = mperp * pyy + eps_n * (2.0*rand_unit() - 1.0);
        mz0 += eps_n * (2.0*rand_unit() - 1.0);

        const double n = sqrt(mx0*mx0 + my0*my0 + mz0*mz0);
        if (n > 1.0e-12) { mx0/=n; my0/=n; mz0/=n; }

        ydata[mx_i] = SUN_RCONST(mx0);
        ydata[my_i] = SUN_RCONST(my0);
        ydata[mz_i] = SUN_RCONST(mz0);
        abstol_data[mx_i] = ATOL1;
        abstol_data[my_i] = ATOL2;
        abstol_data[mz_i] = ATOL3;
      }
    }
  }
  N_VCopyToDevice_Cuda(y);
  N_VCopyToDevice_Cuda(abstol);

#if ENABLE_OUTPUT
  WriteFrame(fp, T0, nx, ny, ng, ncell, y, udata.h_w);
#endif

  cvode_mem = CVodeCreate(CV_BDF, sunctx);
  if (!cvode_mem) { fprintf(stderr, "CVodeCreate failed.\n"); goto cleanup; }

  CHECK_SUNDIALS(CVodeInit(cvode_mem, f, T0, y));
  CHECK_SUNDIALS(CVodeSetUserData(cvode_mem, &udata));
  CHECK_SUNDIALS(CVodeSVtolerances(cvode_mem, RTOL, abstol));

  NLS = SUNNonlinSol_Newton(y, sunctx);
  if (!NLS) { fprintf(stderr, "SUNNonlinSol_Newton failed.\n"); goto cleanup; }
  CHECK_SUNDIALS(CVodeSetNonlinearSolver(cvode_mem, NLS));

  LS = SUNLinSol_SPGMR(y, SUN_PREC_LEFT, KRYLOV_DIM, sunctx);
  if (!LS) { fprintf(stderr, "SUNLinSol_SPGMR failed.\n"); goto cleanup; }
  CHECK_SUNDIALS(CVodeSetLinearSolver(cvode_mem, LS, NULL));
  CHECK_SUNDIALS(CVodeSetJacTimes(cvode_mem, NULL, JtvProduct));
  CHECK_SUNDIALS(CVodeSetPreconditioner(cvode_mem, PrecondSetup, PrecondSolve));

  if (neq < 500000) {
    CHECK_SUNDIALS(SUNLinSol_SPGMRSetGSType(LS, SUN_CLASSICAL_GS));
    printf("GS type: Classical (overhead-limited, neq=%d)\n", neq);
  } else {
    printf("GS type: Modified (bandwidth-limited, neq=%d)\n", neq);
  }
  CHECK_SUNDIALS(CVodeSetMaxOrd(cvode_mem, MAX_BDF_ORDER));

  printf("\n2D Voronoi polycrystal + dead-grain holes + FFT demag\n");
  printf("LLG form: standard with per-cell easy axis, |n̂|=1 enforced\n");
  printf("nx=%d  ny=%d  ng=%d  ncell=%d  neq=%d\n", nx, ny, ng, ncell, neq);
  printf("DEMAG_STRENGTH=%.4f  DEMAG_THICK=%.4f  T_TOTAL=%.2f  RTOL=%.1e\n",
         dstr, dthk, (double)T_TOTAL, (double)RTOL_VAL);

  NOUT = (long int)(ttotal / T1 + SUN_RCONST(0.5));
  iout = 0;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start, 0));

  while (iout < NOUT) {
    retval = CVode(cvode_mem, tout, y, &t, CV_NORMAL);
    if (retval != CV_SUCCESS) {
      fprintf(stderr, "CVode error at output %ld: retval = %d\n", iout, retval);
      break;
    }
#if ENABLE_OUTPUT
    if (ShouldWriteFrame(iout + 1, t))
      WriteFrame(fp, t, nx, ny, ng, ncell, y, udata.h_w);
#endif
    iout++;
    tout += T1;
  }

  CHECK_CUDA(cudaEventRecord(stop, 0));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("GPU simulation took %.3f ms\n", elapsedTime);
  PrintFinalStats(cvode_mem, LS);

cleanup:
  if (LS) SUNLinSolFree(LS);
  if (NLS) SUNNonlinSolFree(NLS);
  if (cvode_mem) CVodeFree(&cvode_mem);
  if (y) N_VDestroy(y);
  if (abstol) N_VDestroy(abstol);
  if (sunctx) SUNContext_Free(&sunctx);
  Precond_Destroy(udata.pd);
  Demag_Destroy(udata.demag);
  if (udata.d_hdmag)    cudaFree(udata.d_hdmag);
  if (udata.d_w)        cudaFree(udata.d_w);
  if (udata.d_y_eff)    cudaFree(udata.d_y_eff);
  if (udata.d_msk)      cudaFree(udata.d_msk);
  if (udata.d_nsk)      cudaFree(udata.d_nsk);
  if (udata.h_w)        free(udata.h_w);
  if (udata.h_msk)      free(udata.h_msk);
  if (udata.h_nsk)      free(udata.h_nsk);
  if (udata.h_grain_id) free(udata.h_grain_id);
  FusedNVec_FreePool();
#if ENABLE_OUTPUT
  if (fp) fclose(fp);
#endif
  return 0;
}