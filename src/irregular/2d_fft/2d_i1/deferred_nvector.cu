/*
 * deferred_nvector.cu  —  v4: multi_dot only
 *
 * ─── What changed from v3 ────────────────────────────────────────────
 * linear_comb_kernel REMOVED.
 *
 * Nsight Systems profiling at neq=196608 showed linear_comb_kernel was
 * the #1 GPU bottleneck at 31.6% of total kernel time (4.59 s, 114,887
 * calls, 40 µs avg).  Root cause: reading K≥5 arrays of 1.5 MB each in
 * a single kernel pass causes L2 cache thrashing across 128 SMs.
 *
 * SUNDIALS' fallback — sequential N_VLinearSum calls (2 arrays per call,
 * ~2.5 µs each) — is 3× faster in aggregate because each call fits in
 * L2 comfortably.  Disabling linear_comb saves ~3 s per run (~20% of
 * total GPU kernel time).
 *
 * The threshold-based regime logic (v3) was too aggressive: the
 * crossover is not at neq=500K as originally estimated but much lower
 * (~50K on Ada Lovelace, depending on Krylov dim and L2 partitioning).
 * Rather than guess a new threshold, we simply never install the
 * override — the multi-array fused kernel has no regime where it wins
 * over SUNDIALS' sequential 2-vector approach on current hardware.
 *
 * ─── What is still overridden ────────────────────────────────────────
 * - nvdotprodmulti  : K dot products in 1 kernel + 1 sync.
 *                     Profile shows 9.5% (1.38 s) for 77K calls.
 *                     Without it (MGS with per-dot sync), cost would be
 *                     ~16 s (386K individual dots × 43 µs each).
 *                     This is a 12× win — the clearest optimization.
 *
 * - nvclone         : propagates multi_dot to all CVODE-internal vectors.
 *
 * ─── Important: N_VLinearCombination fallback ────────────────────────
 * N_VEnableFusedOps_Cuda(v, 1) sets ALL fused op pointers, including
 * SUNDIALS' own nvlinearcombination which has the same K-array problem.
 * We explicitly NULL it out after the enable call so SUNDIALS falls back
 * to the sequential N_VScale + N_VLinearSum loop.  This is the key fix.
 *
 * ─── Pairing with Classical GS ──────────────────────────────────────
 * CGS batches all K dot products in GMRES orthogonalization into one
 * N_VDotProdMulti call.  With our override that's 1 kernel + 1 sync
 * instead of K kernels + K syncs.  Keep SUN_CLASSICAL_GS in main.
 */

#include "deferred_nvector.h"

#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

/* Compile-time constants */
#define FUSED_MAX_VECS   16
#define FUSED_BLOCK_SIZE 256

/* Kernel argument bundles — passed by VALUE into constant cache,
 * no H2D memcpy needed. */
typedef struct { const sunrealtype *p[FUSED_MAX_VECS]; } FusedPtrBundle;

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

/* Kernel: multi_dot_kernel
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

/* Helper */
static int grid_for(sunindextype n)
{
    int g = (int)((n + FUSED_BLOCK_SIZE - 1) / FUSED_BLOCK_SIZE);
    return (g > 65535) ? 65535 : g;
}

/* forward decl */
static N_Vector FusedNVec_Clone(N_Vector w);
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

static N_Vector FusedNVec_Clone(N_Vector w)
{
    N_Vector v = g_original_clone(w);
    if (!v) return NULL;

    /* Tier 1: enable SUNDIALS built-in fused ops */
    N_VEnableFusedOps_Cuda(v, 1);

    /* Tier 3: our multi_dot override */
    v->ops->nvdotprodmulti = FusedNVec_DotProdMulti;

    /* CRITICAL: NULL out linear combination to force sequential fallback.
     * N_VEnableFusedOps_Cuda sets it to SUNDIALS' fused kernel which has
     * the same K-array L2 thrashing problem as our custom kernel. */
    v->ops->nvlinearcombination = NULL;

    /* Propagate clone override */
    v->ops->nvclone = FusedNVec_Clone;

    return v;
}

/* Public entry point */
void FusedNVec_Init(N_Vector v)
{
    if (!v) return;

    sunindextype neq = N_VGetLength(v);

    if (!g_original_clone)
        g_original_clone = v->ops->nvclone;

    /* Tier 1: SUNDIALS built-in fused ops */
    N_VEnableFusedOps_Cuda(v, 1);

    /* Tier 3: custom multi_dot (big win with CGS) */
    v->ops->nvdotprodmulti = FusedNVec_DotProdMulti;

    /* Force sequential fallback for N_VLinearCombination.
     * Profile-proven: SUNDIALS' 2-vector N_VLinearSum loop is 3× faster
     * than any K-vector fused kernel at this problem size. */
    v->ops->nvlinearcombination = NULL;

    /* Propagate to all CVODE-internal clones */
    v->ops->nvclone = FusedNVec_Clone;

    ensure_pool();

    printf("[FusedNVec v4] neq=%ld: multi_dot override active.\n"
           "              linear_comb DISABLED (profile: L2 thrashing at K>=5).\n"
           "              N_VLinearCombination -> sequential N_VLinearSum fallback.\n"
           "              Use Classical GS for multi_dot benefit.\n",
           (long)neq);
}
