#ifndef FUSED_NVEC_H
#define FUSED_NVEC_H

/* Tier 1 + Tier 3 optimization layer on top of SUNDIALS NVECTOR_CUDA.
 * See deferred_nvector.cu for full description.
 */

#include <sundials/sundials_nvector.h>
#include <nvector/nvector_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

void FusedNVec_Init(N_Vector v);
void FusedNVec_FreePool(void);

#ifdef __cplusplus
}
#endif

#endif /* FUSED_NVEC_H */
