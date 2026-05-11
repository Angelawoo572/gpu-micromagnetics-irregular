/**
 * 2D LLG spin-wave single-slit diffraction
 * CVODE + CUDA, SoA layout, ymsk geometry, FFT demag
 *
 * ─── Concept ────────────────────────────────────────────────────────
 * Spin waves (magnons) propagate through a ferromagnetic slab.
 * A vertical strip of hole cells acts as a non-magnetic screen with a
 * single slit opening.  A sinusoidal mz perturbation on the left drives
 * spin waves that diffract through the slit — producing the same
 * single-slit diffraction pattern as an EM wave through a PEC screen,
 * but governed by the LLG equation with FFT demag.
 *
 * ─── Geometry ───────────────────────────────────────────────────────
 *
 *   col:  0        SRC_COL   SCREEN_COL            ng-1
 *         |           |           |                  |
 *         [  active   | src strip |  screen (hole)   |  active (observation) ]
 *                                 | except slit rows |
 *
 *   The screen is a vertical column of hole cells at SCREEN_COL.
 *   The slit is a gap of SLIT_W active cells centered at ny/2.
 *   SRC_COL is a column inside the left active region where we inject
 *   a sinusoidal mz perturbation as a "soft source".
 *
 * ─── Physics ────────────────────────────────────────────────────────
 * Standard simplified LLG:
 *   dm/dt = γ(m × h_eff) + α(h_eff − (m·h_eff)m)
 *
 * h_eff = h_exchange + h_anisotropy + h_DMI + h_demag + h_source
 *
 * h_source: added only at SRC_COL as a soft driving field
 *   h_src_z(t) = H_drive * sin(2π f_drive t) * ramp(t)
 *
 * This tilts mz at the source column, launching spin waves.
 *
 * ─── State vector ───────────────────────────────────────────────────
 * SoA, length 3*ncell:
 *   [0..ncell-1]       mx
 *   [ncell..2ncell-1]  my
 *   [2ncell..3ncell-1] mz
 *
 * ─── ymsk ───────────────────────────────────────────────────────────
 * ymsk = 1 on active cells, 0 on hole (screen) cells.
 * Hole cells: m=0 always, yd=0 via mask multiplication.
 * No branching — identical to i2 style.
 *
 * ─── Output ─────────────────────────────────────────────────────────
 * Binary file output.bin: frames of mz(x,y,t) + header.
 * Use plot_slit_spinwave.py to visualize the diffraction pattern.
 */

#include <cvode/cvode.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <nvector/nvector_cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>
#include <sundials/sundials_iterative.h>

#include "precond.h"
#include "jtv.h"
#include "demag_fft.h"

/* ─── Build knobs (override with -D on compile line) ─────────────── */
#ifndef NX_VAL
#define NX_VAL 768
#endif
#ifndef NY_VAL
#define NY_VAL 256
#endif
#ifndef GROUPSIZE
#define GROUPSIZE 3
#endif

/* Screen / slit geometry (fractions of grid) */
#ifndef SCREEN_X_FRAC
#define SCREEN_X_FRAC  0.50   /* screen at 50% of ng */
#endif
#ifndef SLIT_W_FRAC
#define SLIT_W_FRAC    0.10   /* slit width = 10% of ny */
#endif
#ifndef SRC_X_FRAC
#define SRC_X_FRAC     0.25   /* source column at 25% of ng */
#endif

/* Spin-wave source */
#ifndef DRIVE_FREQ
#define DRIVE_FREQ     0.12
#endif
#ifndef H_DRIVE
#define H_DRIVE        0.10
#endif
#ifndef RAMP_TIME
#define RAMP_TIME      30.0
#endif

/* LLG physics */
#ifndef C_ALPHA
#define C_ALPHA   0.005
#endif
#ifndef C_CHE
#define C_CHE     4.0         /* exchange stiffness */
#endif
#ifndef C_CHG
#define C_CHG     1.0         /* gyromagnetic ratio */
#endif
#ifndef C_CHK
#define C_CHK     0.5         /* uniaxial anisotropy */
#endif
#ifndef C_CHB
#define C_CHB     0.1         /* DMI */
#endif
#ifndef DEMAG_STRENGTH
#define DEMAG_STRENGTH 1.0
#endif
#ifndef DEMAG_THICK
#define DEMAG_THICK    1.0
#endif

/* Absorbing boundary layer (PML-style damping sponge).
 * The last PML_CELLS columns on the LEFT  (x < PML_CELLS)
 * and the last PML_CELLS columns on the RIGHT (x > ng-1-PML_CELLS)
 * get c_alpha ramped up to PML_ALPHA_MAX.
 * This kills reflected spin waves before they re-enter the domain.  */
#ifndef PML_CELLS
#define PML_CELLS      20
#endif
#ifndef PML_ALPHA_MAX
#define PML_ALPHA_MAX  0.5   /* gentle — avoids stiffness spike */
#endif

/* Solver */
#ifndef RTOL_VAL
#define RTOL_VAL   1.0e-4
#endif
#ifndef ATOL_VAL
#define ATOL_VAL   1.0e-6
#endif
#ifndef KRYLOV_DIM
#define KRYLOV_DIM 5
#endif
#ifndef MAX_BDF_ORDER
#define MAX_BDF_ORDER 5
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef BLOCK_X
#define BLOCK_X 16
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 16
#endif

/* Time integration */
#ifndef T_TOTAL
#define T_TOTAL     500.0
#endif
#ifndef T1
#define T1          5.0       /* output interval */
#endif
#ifndef EARLY_SAVE_UNTIL
#define EARLY_SAVE_UNTIL 50.0
#endif
#ifndef EARLY_SAVE_EVERY
#define EARLY_SAVE_EVERY 1
#endif
#ifndef LATE_SAVE_EVERY
#define LATE_SAVE_EVERY  5
#endif
#ifndef ENABLE_OUTPUT
#define ENABLE_OUTPUT 1
#endif
#ifndef PRINT
#define PRINT 0   /* set PRINT=1 to enable console output */
#endif

/* ─── Macros ─────────────────────────────────────────────────────── */
#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__, \
                cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

#define CHECK_SUNDIALS(call) do { \
    int _r = (call); \
    if (_r < 0) { \
        fprintf(stderr,"SUNDIALS error %s:%d: retval=%d\n", \
                __FILE__,__LINE__,_r); \
        exit(1); \
    } \
} while(0)

#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)

/* ─── SoA index helpers ──────────────────────────────────────────── */
__host__ __device__ static inline int idx_mx(int c, int nc) { return c; }
__host__ __device__ static inline int idx_my(int c, int nc) { return nc + c; }
__host__ __device__ static inline int idx_mz(int c, int nc) { return 2*nc + c; }

/* Periodic wrap */
__device__ static inline int wrap_x(int x, int ng) {
    return (x < 0) ? x+ng : (x >= ng ? x-ng : x);
}
__device__ static inline int wrap_y(int y, int ny) {
    return (y < 0) ? y+ny : (y >= ny ? y-ny : y);
}

/* ─── UserData ───────────────────────────────────────────────────── */
typedef struct {
    void           *pd_opaque;    /* PrecondData* */
    sunrealtype    *d_hdmag;      /* FFT demag field, SoA 3*ncell */
    sunrealtype    *d_ymsk;       /* geometry mask, SoA 3*ncell   */
    DemagData      *demag;
    int   nx, ny, ng, ncell, neq;
    int   screen_col;             /* column index of screen */
    int   slit_lo, slit_hi;       /* slit row range [lo, hi) */
    int   src_col;                /* source column index */
    double nxx0, nyy0, nzz0;     /* demag self-coupling */
    double omega_drive;           /* 2π * DRIVE_FREQ */
} UserData;

/* Forward declarations */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data);

/* ─── Smooth ramp ────────────────────────────────────────────────── */
__device__ static inline double src_ramp(double t, double t_ramp) {
    if (t >= t_ramp) return 1.0;
    double x = t / t_ramp;
    return x * x * (3.0 - 2.0 * x);  /* smoothstep */
}

/* ─── normalize_m_kernel (regularized, handles hole cells) ──────── */
__global__ static void normalize_m_kernel(sunrealtype* __restrict__ y, int ncell)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= ncell) return;
    const sunrealtype m1 = y[idx_mx(cell,ncell)];
    const sunrealtype m2 = y[idx_my(cell,ncell)];
    const sunrealtype m3 = y[idx_mz(cell,ncell)];
    const sunrealtype ymp = sqrt(m1*m1 + m2*m2 + m3*m3);
    /* +0.001: hole cells (m=0) stay 0; active cells stay ≈1 */
    const sunrealtype inv = SUN_RCONST(1.0) / (ymp + SUN_RCONST(0.001));
    y[idx_mx(cell,ncell)] = m1 * inv;
    y[idx_my(cell,ncell)] = m2 * inv;
    y[idx_mz(cell,ncell)] = m3 * inv;
}

/* ─── Unified LLG RHS kernel ─────────────────────────────────────── */
__global__ static void f_kernel(
    const sunrealtype* __restrict__ y,
    const sunrealtype* __restrict__ h_dmag,
    const sunrealtype* __restrict__ ymsk,
    sunrealtype*       __restrict__ yd,
    int ng, int ny, int ncell,
    int src_col,
    double omega_drive, double t_now, double ramp_time,
    double h_drive_amp)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= ng || gy >= ny) return;

    const int cell = gy * ng + gx;

    /* Easy-axis anisotropy along z: c_msk = {0,0,1} */
    const sunrealtype c_msk0 = SUN_RCONST(0.0);
    const sunrealtype c_msk1 = SUN_RCONST(0.0);
    const sunrealtype c_msk2 = SUN_RCONST(1.0);
    /* DMI along x: c_nsk = {1,0,0} */
    const sunrealtype c_nsk0 = SUN_RCONST(1.0);
    const sunrealtype c_nsk1 = SUN_RCONST(0.0);
    const sunrealtype c_nsk2 = SUN_RCONST(0.0);

    const sunrealtype c_che   = SUN_RCONST(C_CHE);
    const sunrealtype c_chg   = SUN_RCONST(C_CHG);
    const sunrealtype c_chk   = SUN_RCONST(C_CHK);
    const sunrealtype c_chb   = SUN_RCONST(C_CHB);
    /* PML sponge: ramp alpha from C_ALPHA up to PML_ALPHA_MAX
     * in the first/last PML_CELLS columns (both left and right edges).
     * t_pml in [0,1] at the boundary, 0 in the interior.           */
    sunrealtype c_alpha;
    {
        const int pml = PML_CELLS;
        sunrealtype t_pml = SUN_RCONST(0.0);
        if (gx < pml) {
            /* left absorbing layer */
            t_pml = SUN_RCONST(1.0) - (sunrealtype)gx / (sunrealtype)pml;
        } else if (gx >= ng - pml) {
            /* right absorbing layer */
            t_pml = (sunrealtype)(gx - (ng - pml)) / (sunrealtype)pml;
        }
        /* quadratic ramp: gentle near interior, strong at boundary */
        /* cubic ramp: zero derivative at interior edge → smooth stiffness */
        c_alpha = SUN_RCONST(C_ALPHA)
                + (SUN_RCONST(PML_ALPHA_MAX) - SUN_RCONST(C_ALPHA))
                  * t_pml * t_pml * t_pml;
    }

    const int mx = idx_mx(cell, ncell);
    const int my = idx_my(cell, ncell);
    const int mz = idx_mz(cell, ncell);

    const sunrealtype m1 = y[mx];
    const sunrealtype m2 = y[my];
    const sunrealtype m3 = y[mz];

    /* Neighbors (periodic) */
    const int xl = wrap_x(gx-1, ng), xr = wrap_x(gx+1, ng);
    const int yu = wrap_y(gy-1, ny), yd_idx = wrap_y(gy+1, ny);

    const int lc = gy*ng+xl, rc = gy*ng+xr;
    const int uc = yu*ng+gx, dc = yd_idx*ng+gx;

    const sunrealtype y1L = y[idx_mx(lc,ncell)], y1R = y[idx_mx(rc,ncell)];
    const sunrealtype y1U = y[idx_mx(uc,ncell)], y1D = y[idx_mx(dc,ncell)];
    const sunrealtype y2L = y[idx_my(lc,ncell)], y2R = y[idx_my(rc,ncell)];
    const sunrealtype y2U = y[idx_my(uc,ncell)], y2D = y[idx_my(dc,ncell)];
    const sunrealtype y3L = y[idx_mz(lc,ncell)], y3R = y[idx_mz(rc,ncell)];
    const sunrealtype y3U = y[idx_mz(uc,ncell)], y3D = y[idx_mz(dc,ncell)];

    /* Spin-wave soft source at src_col:
     * Drive h1 (transverse, along x) — this excites mx oscillations
     * which propagate as spin waves via the LLG cross product.
     * Driving h3 (longitudinal, along z=easy axis) causes only
     * uniform precession, not a propagating wave. */
    sunrealtype h_src = ZERO;
    if (gx == src_col) {
        const double ramp = src_ramp((double)t_now, (double)ramp_time);
        h_src = (sunrealtype)(h_drive_amp * ramp * sin(omega_drive * (double)t_now));
    }

    /* Effective field: exchange + anisotropy + DMI + demag + source */
    const sunrealtype h1 =
        c_che*(y1L+y1R+y1U+y1D) +
        c_msk0 * c_chk * m1*(m1*m1 - ONE) +
        c_chb * c_nsk0 * (y1L+y1R) +
        h_dmag[mx] +
        h_src;                          /* <-- transverse driving here */

    const sunrealtype h2 =
        c_che*(y2L+y2R+y2U+y2D) +
        c_msk1 * c_chk * m2*(m2*m2 - ONE) +
        c_chb * c_nsk1 * (y2L+y2R) +
        h_dmag[my];

    const sunrealtype h3 =
        c_che*(y3L+y3R+y3U+y3D) +
        c_msk2 * c_chk * m3*(m3*m3 - ONE) +
        c_chb * c_nsk2 * (y3L+y3R) +
        h_dmag[mz];

    /* LLG */
    const sunrealtype mh = m1*h1 + m2*h2 + m3*h3;

    /* Mask: hole cells → 0 */
    yd[mx] = ymsk[mx] * (c_chg*(m3*h2 - m2*h3) + c_alpha*(h1 - mh*m1));
    yd[my] = ymsk[my] * (c_chg*(m1*h3 - m3*h1) + c_alpha*(h2 - mh*m2));
    yd[mz] = ymsk[mz] * (c_chg*(m2*h1 - m1*h2) + c_alpha*(h3 - mh*m3));
}

/* ─── CVODE RHS wrapper ──────────────────────────────────────────── */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data)
{
    UserData *udata = (UserData*)user_data;
    sunrealtype *ydata    = N_VGetDeviceArrayPointer_Cuda(y);
    sunrealtype *ydotdata = N_VGetDeviceArrayPointer_Cuda(ydot);

    /* Step 0: normalize (regularized) */
    {
        const int b = BLOCK_SIZE;
        const int g = (udata->ncell + b - 1) / b;
        normalize_m_kernel<<<g, b>>>(ydata, udata->ncell);
    }

    /* Step 1: FFT demag */
    if (udata->demag && DEMAG_STRENGTH > 0.0) {
        Demag_Apply(udata->demag,
                    (const double*)ydata,
                    (double*)udata->d_hdmag);
    } else {
        cudaMemsetAsync(udata->d_hdmag, 0,
                        (size_t)3 * udata->ncell * sizeof(sunrealtype), 0);
    }

    /* Step 2: LLG RHS with source */
    {
        dim3 block(BLOCK_X, BLOCK_Y);
        dim3 grid((udata->ng + BLOCK_X - 1) / BLOCK_X,
                  (udata->ny + BLOCK_Y - 1) / BLOCK_Y);
        f_kernel<<<grid, block>>>(
            ydata, udata->d_hdmag, udata->d_ymsk, ydotdata,
            udata->ng, udata->ny, udata->ncell,
            udata->src_col,
            udata->omega_drive, (double)t, (double)RAMP_TIME,
            (double)H_DRIVE);
    }

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "f kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

/* ─── Output helpers ─────────────────────────────────────────────── */
#if ENABLE_OUTPUT
static int g_frame_count = 0;

static int ShouldWriteFrame(long iout, sunrealtype t)
{
    if (t <= (sunrealtype)EARLY_SAVE_UNTIL)
        return ((iout % EARLY_SAVE_EVERY) == 0);
    else
        return ((iout % LATE_SAVE_EVERY) == 0);
}

static void WriteFrame(FILE *fp, sunrealtype t,
                       int nx, int ny, int ng, int ncell, N_Vector y)
{
    N_VCopyFromDevice_Cuda(y);
    sunrealtype *ydata = N_VGetHostArrayPointer_Cuda(y);

    /* Text format matching other LLG codes:
     *   # t=<time>
     *   mx my mz    (one line per cell, row-major j*ng+i)            */
    fprintf(fp, "# t=%.6f nx=%d ny=%d ng=%d\n", (double)t, nx, ny, ng);
    for (int cell = 0; cell < ncell; cell++) {
        fprintf(fp, "%.8f %.8f %.8f\n",
                (double)ydata[idx_mx(cell, ncell)],
                (double)ydata[idx_my(cell, ncell)],
                (double)ydata[idx_mz(cell, ncell)]);
    }
    g_frame_count++;
}
#endif

static void PrintFinalStats(void *cvode_mem)
{
    long int nst, nfe, nsetups, nni, ncfn, netf, nli, nlcf, njvevals;
    CVodeGetNumSteps(cvode_mem, &nst);
    CVodeGetNumRhsEvals(cvode_mem, &nfe);
    CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
    CVodeGetNumErrTestFails(cvode_mem, &netf);
    CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
    CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
    CVodeGetNumLinIters(cvode_mem, &nli);
    CVodeGetNumLinConvFails(cvode_mem, &nlcf);
    CVodeGetNumJtimesEvals(cvode_mem, &njvevals);
    printf("\nFinal Statistics:\n");
    printf("nst=%-6ld nfe=%-6ld nsetups=%-6ld nni=%-6ld ncfn=%-6ld netf=%-6ld\n",
           nst, nfe, nsetups, nni, ncfn, netf);
    printf("nli=%-6ld nlcf=%-6ld njvevals=%ld\n", nli, nlcf, njvevals);
}

/* ─── main ───────────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
    (void)argc; (void)argv;

    /* Grid */
    const int nx    = NX_VAL;
    const int ny    = NY_VAL;
    if (nx % GROUPSIZE != 0) {
        fprintf(stderr, "NX_VAL must be divisible by GROUPSIZE=%d\n", GROUPSIZE);
        return 1;
    }
    const int ng    = nx / GROUPSIZE;
    const int ncell = ng * ny;
    const int neq   = 3 * ncell;

    /* Geometry */
    const int screen_col = (int)(SCREEN_X_FRAC * (double)ng + 0.5);
    const int slit_w     = (int)(SLIT_W_FRAC   * (double)ny + 0.5);
    const int slit_lo    = ny/2 - slit_w/2;
    const int slit_hi    = slit_lo + slit_w;
    const int src_col    = (int)(SRC_X_FRAC    * (double)ng + 0.5);

#if PRINT
    printf("=== 2D LLG Spin-Wave Single-Slit Diffraction ===\n");
    printf("  grid         : %d x %d  (ng=%d)\n", nx, ny, ng);
    printf("  screen_col   : %d\n", screen_col);
    printf("  slit rows    : [%d, %d)  width=%d\n", slit_lo, slit_hi, slit_w);
    printf("  src_col      : %d\n", src_col);
    printf("  drive_freq   : %.4f  omega=%.4f\n",
           (double)DRIVE_FREQ, 2.0*3.14159265358979*DRIVE_FREQ);
    printf("  H_drive      : %.4f  ramp_time=%.1f\n", (double)H_DRIVE, (double)RAMP_TIME);
    printf("  alpha        : %.4f\n", (double)C_ALPHA);
    printf("  T_TOTAL      : %.1f   RTOL=%.1e\n", (double)T_TOTAL, (double)RTOL_VAL);
#endif

    /* SUNDIALS context */
    SUNContext sunctx;
    CHECK_SUNDIALS(SUNContext_Create(SUN_COMM_NULL, &sunctx));

    /* UserData */
    UserData udata;
    memset(&udata, 0, sizeof(udata));
    udata.nx         = nx;
    udata.ny         = ny;
    udata.ng         = ng;
    udata.ncell      = ncell;
    udata.neq        = neq;
    udata.screen_col = screen_col;
    udata.slit_lo    = slit_lo;
    udata.slit_hi    = slit_hi;
    udata.src_col    = src_col;
    udata.omega_drive = 2.0 * 3.14159265358979323846 * DRIVE_FREQ;

    /* Preconditioner */
    udata.pd_opaque = (void*)Precond_Create(ng, ny, ncell);
    if (!udata.pd_opaque) { fprintf(stderr, "Precond_Create failed\n"); return 1; }

    /* FFT demag */
    CHECK_CUDA(cudaMalloc((void**)&udata.d_hdmag,
                          (size_t)3*ncell*sizeof(sunrealtype)));
    CHECK_CUDA(cudaMemset(udata.d_hdmag, 0,
                          (size_t)3*ncell*sizeof(sunrealtype)));

    if (DEMAG_STRENGTH > 0.0) {
        udata.demag = Demag_Init(ng, ny, DEMAG_THICK, DEMAG_STRENGTH);
        if (!udata.demag) { fprintf(stderr, "Demag_Init failed\n"); return 1; }
        Demag_GetSelfCoupling(udata.demag,
                              &udata.nxx0, &udata.nyy0, &udata.nzz0);
    #if PRINT
    printf("  demag nxx0=%.4e nyy0=%.4e nzz0=%.4e\n",
               udata.nxx0, udata.nyy0, udata.nzz0);
#endif
    }

    /* Build ymsk: screen column = hole, except slit rows */
    CHECK_CUDA(cudaMalloc((void**)&udata.d_ymsk,
                          (size_t)3*ncell*sizeof(sunrealtype)));
    {
        sunrealtype *h_ymsk = (sunrealtype*)malloc(
            (size_t)3*ncell*sizeof(sunrealtype));
        if (!h_ymsk) { fprintf(stderr, "malloc ymsk failed\n"); return 1; }

        long n_active = 0, n_hole = 0;
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < ng; i++) {
                const int cell = j*ng + i;
                /* Hole = screen column, excluding the slit gap */
                int in_screen = (i == screen_col) &&
                                !((j >= slit_lo) && (j < slit_hi));
                sunrealtype m = in_screen ? ZERO : ONE;
                if (in_screen) n_hole++; else n_active++;
                h_ymsk[idx_mx(cell,ncell)] = m;
                h_ymsk[idx_my(cell,ncell)] = m;
                h_ymsk[idx_mz(cell,ncell)] = m;
            }
        }
        CHECK_CUDA(cudaMemcpy(udata.d_ymsk, h_ymsk,
                              (size_t)3*ncell*sizeof(sunrealtype),
                              cudaMemcpyHostToDevice));
    #if PRINT
    printf("  active=%ld  hole=%ld  (screen col %d, slit [%d,%d))\n",
               n_active, n_hole, screen_col, slit_lo, slit_hi);
#endif
        free(h_ymsk);
    }

    /* Allocate y, abstol */
    N_Vector y      = N_VNew_Cuda(neq, sunctx);
    N_Vector abstol = N_VNew_Cuda(neq, sunctx);
    if (!y || !abstol) { fprintf(stderr, "N_VNew_Cuda failed\n"); return 1; }

    /* Initial condition: m = (0, 0, 1) everywhere (easy axis = z) */
    {
        sunrealtype *ydata       = N_VGetHostArrayPointer_Cuda(y);
        sunrealtype *abstol_data = N_VGetHostArrayPointer_Cuda(abstol);
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < ng; i++) {
                const int cell = j*ng + i;
                ydata[idx_mx(cell,ncell)] = ZERO;
                ydata[idx_my(cell,ncell)] = ZERO;
                ydata[idx_mz(cell,ncell)] = ONE;
                abstol_data[idx_mx(cell,ncell)] = SUN_RCONST(ATOL_VAL);
                abstol_data[idx_my(cell,ncell)] = SUN_RCONST(ATOL_VAL);
                abstol_data[idx_mz(cell,ncell)] = SUN_RCONST(ATOL_VAL);
            }
        }
        /* Zero hole cells */
        /* (ymsk will keep them at 0 via RHS masking) */
        N_VCopyToDevice_Cuda(y);
        N_VCopyToDevice_Cuda(abstol);
    }

    /* CVODE setup */
    void *cvode_mem = CVodeCreate(CV_BDF, sunctx);
    if (!cvode_mem) { fprintf(stderr, "CVodeCreate failed\n"); return 1; }

    CHECK_SUNDIALS(CVodeInit(cvode_mem, f, SUN_RCONST(0.0), y));
    CHECK_SUNDIALS(CVodeSetUserData(cvode_mem, &udata));
    CHECK_SUNDIALS(CVodeSVtolerances(cvode_mem,
                                     SUN_RCONST(RTOL_VAL), abstol));

    SUNNonlinearSolver NLS = SUNNonlinSol_Newton(y, sunctx);
    CHECK_SUNDIALS(CVodeSetNonlinearSolver(cvode_mem, NLS));

    SUNLinearSolver LS = SUNLinSol_SPGMR(y, SUN_PREC_LEFT, KRYLOV_DIM, sunctx);
    CHECK_SUNDIALS(CVodeSetLinearSolver(cvode_mem, LS, NULL));
    CHECK_SUNDIALS(CVodeSetJacTimes(cvode_mem, NULL, JtvProduct));
    CHECK_SUNDIALS(CVodeSetPreconditioner(cvode_mem, PrecondSetup, PrecondSolve));

    if (neq < 500000)
        CHECK_SUNDIALS(SUNLinSol_SPGMRSetGSType(LS, SUN_CLASSICAL_GS));

    CHECK_SUNDIALS(CVodeSetMaxOrd(cvode_mem, MAX_BDF_ORDER));
    CHECK_SUNDIALS(CVodeSetMaxNumSteps(cvode_mem, 100000));
#if PRINT
    printf("  Max BDF order=%d  Krylov dim=%d\n\n", MAX_BDF_ORDER, KRYLOV_DIM);
#endif

#if ENABLE_OUTPUT
    FILE *fp = fopen("output.txt", "w");
    if (!fp) { fprintf(stderr, "Cannot open output.txt\n"); return 1; }
    setvbuf(fp, NULL, _IOFBF, 1 << 20);
    /* File header: geometry info as comments */
    fprintf(fp, "# slit spinwave output\n");
    fprintf(fp, "# nx=%d ny=%d ng=%d\n", nx, ny, ng);
    fprintf(fp, "# screen_col=%d slit_lo=%d slit_hi=%d src_col=%d\n",
            screen_col, slit_lo, slit_hi, src_col);
    /* Write initial frame */
    WriteFrame(fp, SUN_RCONST(0.0), nx, ny, ng, ncell, y);
#endif

    /* Time loop */
    cudaEvent_t tstart, tstop; float ms = 0.0f;
    CHECK_CUDA(cudaEventCreate(&tstart));
    CHECK_CUDA(cudaEventCreate(&tstop));
    CHECK_CUDA(cudaEventRecord(tstart, 0));

    const long int NOUT = (long int)(T_TOTAL / T1 + 0.5);
    sunrealtype t = SUN_RCONST(0.0);

    for (long int iout = 1; iout <= NOUT; iout++) {
        sunrealtype tout = (sunrealtype)((double)iout * T1);
        int retval = CVode(cvode_mem, tout, y, &t, CV_NORMAL);
        if (retval != CV_SUCCESS) {
            fprintf(stderr, "CVode error at iout=%ld retval=%d\n", iout, retval);
            break;
        }
#if ENABLE_OUTPUT
        if (ShouldWriteFrame(iout, t))
            WriteFrame(fp, t, nx, ny, ng, ncell, y);
#endif
#if PRINT
        if (iout % 10 == 0 || iout == NOUT)
            printf("  t = %7.1f / %.1f   (frame %ld)\n",
                   (double)t, (double)T_TOTAL, iout);
#endif
    }

    CHECK_CUDA(cudaEventRecord(tstop, 0));
    CHECK_CUDA(cudaEventSynchronize(tstop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, tstart, tstop));
#if PRINT
    printf("\nGPU simulation took %.3f ms\n", ms);
#endif

#if PRINT
    PrintFinalStats(cvode_mem);
#endif

#if ENABLE_OUTPUT
    fclose(fp);
#if PRINT
    printf("Wrote output.txt (%d frames).  Run plot_slit_spinwave.py\n",
           g_frame_count);
#endif
#endif

    /* Cleanup */
    SUNLinSolFree(LS);
    SUNNonlinSolFree(NLS);
    CVodeFree(&cvode_mem);
    N_VDestroy(y);
    N_VDestroy(abstol);
    SUNContext_Free(&sunctx);
    Precond_Destroy((PrecondData*)udata.pd_opaque);
    if (udata.d_hdmag) cudaFree(udata.d_hdmag);
    if (udata.d_ymsk)  cudaFree(udata.d_ymsk);
    if (udata.demag)   Demag_Destroy(udata.demag);

    return 0;
}
