#ifndef FUSED_NVEC_H
#define FUSED_NVEC_H

/* Tier 1 + Tier 3 optimization layer on top of SUNDIALS NVECTOR_CUDA.
 *
 * Tier 1 (inside FusedNVec_Init):
 *   Calls N_VEnableFusedOps_Cuda(v, 1) to activate SUNDIALS' own
 *   built-in fused ops (requires SUNDIALS built with
 *   -DSUNDIALS_ENABLE_PACKAGE_FUSED_KERNELS=ON for full CVODE-level fusion;
 *   the NVector-level fused ops work regardless of that flag).
 *
 * Tier 3 (custom kernel overrides):
 *   Overrides three fused op function pointers with hand-tuned CUDA kernels
 *   that do more work per synchronization barrier:
 *
 *   - nvdotprodmulti  : K dot products in ONE kernel launch + ONE sync.
 *                       Default SUNDIALS CUDA calls K separate kernels,
 *                       each blocking the CPU.  In SPGMR Gram-Schmidt this
 *                       is called in a tight loop; batching it collapses K
 *                       cudaStreamSynchronize events into 1.
 *
 *   - nvlinearcombination : nv-vector linear combination z = sum c_j * x_j
 *                           in a single kernel, one pass over memory.
 *
 *   - nvscaleaddmulti : nvec simultaneous updates Z[j] = a[j]*x + Y[j]
 *                       in a single kernel, one pass over x.
 *
 *   - nvclone         : ensures every vector CVODE clones internally also
 *                       carries the Tier 1 + Tier 3 ops (propagation).
 *
 * Usage (in main, after N_VNew_Cuda):
 *   FusedNVec_Init(y);          // install on solution vector
 *   ...
 *   FusedNVec_FreePool();       // call once before program exits
 *
 * Correctness guarantee:
 *   All overrides compute mathematically identical results to the originals.
 *   The only change is reduced kernel-launch count and sync count.
 */

#include <sundials/sundials_nvector.h>
#include <nvector/nvector_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * FusedNVec_Init
 *
 * Install Tier 1 + Tier 3 fused ops on an NVECTOR_CUDA object.
 * Also overrides nvclone so that every vector CVODE creates internally
 * automatically inherits the same optimization.
 *
 * Call once, on the primary solution vector y, after N_VNew_Cuda.
 * Do NOT call on abstol (it is not cloned for arithmetic work).
 */
void FusedNVec_Init(N_Vector v);

/*
 * FusedNVec_FreePool
 *
 * Release the small persistent device/pinned-host buffers used by the
 * custom kernels.  Call once at program exit (before CUDA context teardown).
 */
void FusedNVec_FreePool(void);

#ifdef __cplusplus
}
#endif

#endif /* FUSED_NVEC_H */
