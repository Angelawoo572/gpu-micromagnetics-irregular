/*
 * 1. Pointer arrays and coefficient arrays are passed DIRECTLY in the kernel
 *    argument space (goes into the GPU constant cache) — eliminates all
 *    per-call H2D cudaMemcpy for those small arrays.  v1 added ~1.6 s of
 *    extra synchronous memcpy time; v2 adds zero.
 *
 * 2. N_VScaleAddMulti override REMOVED.  Profiling showed our kernel was
 *    slower than SUNDIALS' built-in (379 µs vs 307 µs avg).
 *
 * 3. Now pairs with Classical Gram-Schmidt (CGS) in SPGMR.
 *    Root cause why v1 showed no speedup on dot products:
 *      Modified GS (SPGMR default) calls N_VDotProd in a sequential loop —
 *      each call returns a scalar synchronously, K calls = K syncs.
 *      It is inherently sequential: step i+1 needs the result of step i,
 *      so N_VDotProdMulti is NEVER called by MGS.
 *
 *    Classical GS first computes all K dot products against the
 *    un-modified vector, then applies all corrections.  SUNDIALS
 *    dispatches that batch through N_VDotProdMulti when the op
 *    pointer is non-NULL.  Result: K syncs per GMRES iteration → 1 sync.
 *
 *    To activate, add after SUNLinSol_SPGMR(...):
 *      SUNLinSol_SPGMRSetGSType(LS, SUN_CLASSICAL_GS);
 *    (already done in the updated 2d_p.cu)
 *
 * What is still overridden
 * - nvdotprodmulti     : K dot products in 1 kernel + 1 sync (big win w/ CGS)
 * - nvlinearcombination: nv-vector combo in 1 kernel, 1 memory pass
 * - nvclone            : propagates both to all CVODE-internal vectors
 *
 * Why struct-based kernel args work for small arrays
 * CUDA copies kernel arguments into a 4 KB per-launch constant buffer.
 * A struct with 16 double-pointers = 128 bytes fits entirely there and
 * is served from the constant cache — same bandwidth as __constant__,
 * zero extra API calls.
 */

#include "deferred_nvector.h"

#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

/* Compile-time constants */

/* SPGMR default Krylov dim = 5; 16 is generous headroom.
 * Shared memory per block: 16 * 256 * 8 = 32 768 B < 48 KB.  */
#define FUSED_MAX_VECS   16
#define FUSED_BLOCK_SIZE 256

/* Kernel argument bundles — passed by VALUE into constant cache,
 * no H2D memcpy needed. */
typedef struct { const sunrealtype *p[FUSED_MAX_VECS]; } FusedPtrBundle;
typedef struct { sunrealtype        c[FUSED_MAX_VECS]; } FusedCoeffBundle;

/* Persistent device pool — ONLY for reduction output scalars */
static sunrealtype *d_result_pool = NULL;
static sunrealtype *h_result_pool = NULL;
static int          g_pool_ready  = 0;

static void ensure_pool(void)
{
    if (g_pool_ready) return;
    if (cudaMalloc    ((void**)&d_result_pool, FUSED_MAX_VECS * sizeof(sunrealtype)) != cudaSuccess ||
        cudaMallocHost((void**)&h_result_pool, FUSED_MAX_VECS * sizeof(sunrealtype)) != cudaSuccess) {
        fprintf(stderr, "fused_nvec: pool allocation failed\n"); return;
    }
    g_pool_ready = 1;
}

void FusedNVec_FreePool(void)
{
    if (!g_pool_ready) return;
    cudaFree(d_result_pool);
    cudaFreeHost(h_result_pool);
    d_result_pool = NULL;
    h_result_pool = NULL;
    g_pool_ready  = 0;
}

/* Kernel 1: multi_dot_kernel
 *
 * out[k] = sum_i  x[i] * Y.p[k][i]   for k = 0..K-1
 *
 * Y pointers are in the constant-cache parameter bundle —
 * no extra H2D memcpy, no extra API call.
 * x[] read once and reused K times (stays in L1/L2 cache).
 *
 * With Classical GS this collapses K syncs to 1. */
__global__ static void multi_dot_kernel(
    const sunrealtype* __restrict__ x,
    FusedPtrBundle                  Y,
    sunrealtype*       __restrict__ out,
    sunindextype                    n,
    int                             K)
{
    extern __shared__ sunrealtype smem[]; /* K * blockDim.x doubles */

    const int tid    = (int)threadIdx.x;
    const int bsz    = (int)blockDim.x;
    sunindextype gid    = (sunindextype)blockIdx.x * bsz + tid;
    sunindextype stride = (sunindextype)gridDim.x  * bsz;

    for (int k = 0; k < K; k++)
        smem[k * bsz + tid] = SUN_RCONST(0.0);
    __syncthreads();

    for (sunindextype i = gid; i < n; i += stride) {
        const sunrealtype xi = x[i];
        for (int k = 0; k < K; k++)
            smem[k * bsz + tid] += xi * Y.p[k][i];
    }
    __syncthreads();

    for (int s = bsz >> 1; s > 0; s >>= 1) {
        if (tid < s)
            for (int k = 0; k < K; k++)
                smem[k * bsz + tid] += smem[k * bsz + tid + s];
        __syncthreads();
    }

    if (tid == 0)
        for (int k = 0; k < K; k++)
            atomicAdd(&out[k], smem[k * bsz]);
}

/* Kernel 2: linear_comb_kernel
 *
 * z[i] = sum_{j} C.c[j] * X.p[j][i]
 *
 * Coefficients and pointers both in the constant-cache bundle.
 * Single memory pass. */
__global__ static void linear_comb_kernel(
    int              nv,
    FusedCoeffBundle C,
    FusedPtrBundle   X,
    sunrealtype*     __restrict__ z,
    sunindextype     n)
{
    sunindextype i      = (sunindextype)blockIdx.x * blockDim.x + threadIdx.x;
    sunindextype stride = (sunindextype)gridDim.x  * blockDim.x;

    for (; i < n; i += stride) {
        sunrealtype s = SUN_RCONST(0.0);
        for (int j = 0; j < nv; j++)
            s += C.c[j] * X.p[j][i];
        z[i] = s;
    }
}

/* Helper */
static int grid_for(sunindextype n)
{
    int g = (int)((n + FUSED_BLOCK_SIZE - 1) / FUSED_BLOCK_SIZE);
    return (g > 65535) ? 65535 : g;
}

/* forward decl */
static N_Vector FusedNVec_Clone(N_Vector w);
static N_Vector FusedNVec_Clone_impl(N_Vector w, int use_lc);
static N_Vector (*g_original_clone)(N_Vector) = NULL;

/* N_VDotProdMulti override
 * Called by SPGMR Classical GS when nvdotprodmulti != NULL.
 * K × N_VDotProd (K syncs)  →  1 kernel + 1 sync. */
static SUNErrCode FusedNVec_DotProdMulti(
    int nvec, N_Vector x, N_Vector *Y, sunrealtype *dotprods)
{
    if (nvec <= 0) return SUN_SUCCESS;

    if (nvec > FUSED_MAX_VECS) {
        for (int j = 0; j < nvec; j++)
            dotprods[j] = N_VDotProd(x, Y[j]);
        return SUN_SUCCESS;
    }

    ensure_pool();

    const sunindextype  n  = N_VGetLength(x);
    const sunrealtype  *xd = N_VGetDeviceArrayPointer_Cuda(x);

    /* build pointer bundle on the CPU stack — zero memcpy */
    FusedPtrBundle Yb;
    for (int j = 0; j < nvec; j++)
        Yb.p[j] = N_VGetDeviceArrayPointer_Cuda(Y[j]);

    /* async-zero result buffer then launch kernel on stream 0 */
    cudaMemsetAsync(d_result_pool, 0, (size_t)nvec * sizeof(sunrealtype), 0);

    multi_dot_kernel<<<grid_for(n), FUSED_BLOCK_SIZE,
                       (size_t)nvec * FUSED_BLOCK_SIZE * sizeof(sunrealtype)>>>(
        xd, Yb, d_result_pool, n, nvec);

    /* ONE sync + one D2H copy for all K scalars */
    cudaMemcpy(dotprods, d_result_pool,
               (size_t)nvec * sizeof(sunrealtype),
               cudaMemcpyDeviceToHost);   /* implicit sync */

    return SUN_SUCCESS;
}

/* 
 * N_VLinearCombination override
 *
 * z = sum_{j} c[j] * X[j]  in one kernel, one memory pass.
 * Bundles go via kernel arg space — zero H2D overhead. */
static SUNErrCode FusedNVec_LinearCombination(
    int nv, sunrealtype *c, N_Vector *X, N_Vector z)
{
    if (nv <= 0) return SUN_SUCCESS;
    if (nv == 1) { N_VScale(c[0], X[0], z); return SUN_SUCCESS; }

    if (nv > FUSED_MAX_VECS) {
        N_VScale(c[0], X[0], z);
        for (int j = 1; j < nv; j++)
            N_VLinearSum(SUN_RCONST(1.0), z, c[j], X[j], z);
        return SUN_SUCCESS;
    }

    const sunindextype  n  = N_VGetLength(z);
    sunrealtype        *zd = N_VGetDeviceArrayPointer_Cuda(z);

    FusedCoeffBundle Cb;
    FusedPtrBundle   Xb;
    for (int j = 0; j < nv; j++) {
        Cb.c[j] = c[j];
        Xb.p[j] = N_VGetDeviceArrayPointer_Cuda(X[j]);
    }

    linear_comb_kernel<<<grid_for(n), FUSED_BLOCK_SIZE>>>(nv, Cb, Xb, zd, n);

    /* async — no sync needed, result stays on device */
    return SUN_SUCCESS;
}

/*
 * Threshold between overhead-limited and bandwidth-limited regimes.
 *
 * Below this threshold the problem is small enough that CVODE
 * orchestration overhead dominates.  Switching to Classical GS and
 * installing the linear_comb_kernel override reduces sync count and
 * wins on end-to-end time.
 *
 * Above this threshold each kernel is long enough that bandwidth
 * (not sync count) is the bottleneck.  The linear_comb_kernel reads
 * K large arrays simultaneously (K=Krylov dim, default 5), causing
 * L2 cache thrashing and ~17% effective bandwidth — far worse than
 * SUNDIALS' sequential N_VLinearSum calls (88% efficient, 2 arrays
 * per call).  CGS would add 54+ seconds on 3M-element problems.
 * For large n: use Modified GS (the default) and skip the
 * linear_comb override.  Only the multi_dot override is kept so
 * that IF the caller switches to CGS independently, it still helps.
 *
 * Empirical crossover:  ~500 K elements.
 * Adjust if your hardware has a different L2 / bandwidth ratio.
 */
#define FUSED_SMALL_NEQ_THRESHOLD 500000

/* g_use_linear_comb_override is set in FusedNVec_Init based on neq */
static int g_use_lc_override = 0;

static N_Vector FusedNVec_Clone_impl(N_Vector w, int use_lc)
{
    N_Vector v = g_original_clone(w);
    if (!v) return NULL;
    N_VEnableFusedOps_Cuda(v, 1);
    v->ops->nvdotprodmulti = FusedNVec_DotProdMulti;
    if (use_lc)
        v->ops->nvlinearcombination = FusedNVec_LinearCombination;
    v->ops->nvclone = FusedNVec_Clone;
    return v;
}

static N_Vector FusedNVec_Clone(N_Vector w)
{
    return FusedNVec_Clone_impl(w, g_use_lc_override);
}

/* Public entry point
 *
 * neq  : total number of equations (N_VGetLength(v)).
 *        Used to choose the right regime automatically.
 */
void FusedNVec_Init(N_Vector v)
{
    if (!v) return;

    sunindextype neq = N_VGetLength(v);

    if (!g_original_clone)
        g_original_clone = v->ops->nvclone;

    /* Decide regime */
    g_use_lc_override = (neq < FUSED_SMALL_NEQ_THRESHOLD) ? 1 : 0;

    N_VEnableFusedOps_Cuda(v, 1);                         /* Tier 1 */
    v->ops->nvdotprodmulti = FusedNVec_DotProdMulti;      /* Tier 3 */
    if (g_use_lc_override)
        v->ops->nvlinearcombination = FusedNVec_LinearCombination;
    v->ops->nvclone = FusedNVec_Clone;

    ensure_pool();

    if (g_use_lc_override) {
        printf("[FusedNVec v3] neq=%ld < %d: OVERHEAD-LIMITED regime.\n"
               "               Tier1 + multi_dot + linear_comb installed.\n"
               "               Use Classical GS for full benefit.\n",
               (long)neq, FUSED_SMALL_NEQ_THRESHOLD);
    } else {
        printf("[FusedNVec v3] neq=%ld >= %d: BANDWIDTH-LIMITED regime.\n"
               "               Tier1 + multi_dot only (linear_comb skipped).\n"
               "               Keep Modified GS (default) for this size.\n",
               (long)neq, FUSED_SMALL_NEQ_THRESHOLD);
    }
}
