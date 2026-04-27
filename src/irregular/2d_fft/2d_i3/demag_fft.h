#ifndef FFT_DEMAG_H
#define FFT_DEMAG_H

#include <cuda_runtime.h>   /* for cudaStream_t */

/*
 * demag_fft.h  —  FFT demagnetization field, GPU-only pipeline.
 *
 * NEW IN PMPP-TILED EDITION:
 *   - Demag_ApplyWindowed: gather kernel multiplies y by per-cell w on
 *     the fly, eliminating the apply_weight_kernel pre-pass that used
 *     to write a 3*ncell scratch.
 *   - Demag_SetStream:   route all cuFFT calls + scatter onto a
 *     user-supplied CUDA stream so the demag pipeline runs concurrently
 *     with other kernels.
 *
 * Demag_Apply (no-window) is preserved for callers that don't need
 * windowing — same binary as before.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DemagData DemagData;

DemagData* Demag_Init(int nx, int ny, double thick, double demag_strength);

/* Original: full magnetization, no windowing — unchanged ABI. */
void       Demag_Apply(DemagData *d, const double *y_dev, double *h_out_dev);

/* NEW: applies per-cell weight w on the fly during the gather step.
 * Mathematically equivalent to first computing y_eff = w·y on a separate
 * pass and calling Demag_Apply(y_eff), but saves one full 3*ncell pass
 * over global memory.  All pointers must be device pointers. */
void       Demag_ApplyWindowed(DemagData *d,
                               const double *y_dev,
                               const double *w_dev,
                               double *h_out_dev);

/* NEW: route cuFFT execution + helper kernels onto `stream`.  Pass 0
 * (default stream) to revert.  Cheap; safe to call every f(). */
void       Demag_SetStream(DemagData *d, cudaStream_t stream);

void       Demag_Destroy(DemagData *d);

void       Demag_GetSelfCoupling(DemagData *d,
                                 double *nxx0_scaled,
                                 double *nyy0_scaled,
                                 double *nzz0_scaled);

#ifdef __cplusplus
}
#endif
#endif
