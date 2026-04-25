#ifndef FFT_DEMAG_H
#define FFT_DEMAG_H

/*
 * demag_fft.h  —  FFT demagnetization field, GPU-only pipeline.
 *
 * Per f() call, fully on device (D2Z/Z2D real-to-complex, half-spectrum):
 *     gather_real_kernel : SoA y -> 3 contiguous real arrays
 *     cufftExecD2Z       : FORWARD batched (batch=3) -> M̂ half-spectrum
 *     multiply_kernel    : Ĥ = f̂ · M̂
 *     cufftExecZ2D       : INVERSE batched (batch=3) -> H real
 *     scatter_real_kernel: FFT-shift + strength/N scaling -> SoA h_out_dev
 *
 * f̂ is computed ONCE in Demag_Init and stored permanently on device
 * as half-spectrum (9 components × ncell_half complex).  No host
 * transfers during Apply.
 *
 * Self-coupling  (for preconditioner):
 *     N(0) = diag(Nxx(0), Nyy(0), Nzz(0))   (off-diagonals vanish at r=0)
 *     Scaled by `strength` (matching the final h_out scale).
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DemagData DemagData;

DemagData* Demag_Init(int nx, int ny, double thick, double demag_strength);

/* Fully on-GPU: y_dev and h_out_dev are device pointers; h_out_dev is OVERWRITTEN.
 * h_out_dev must be a 3*ncell SoA buffer on device. */
void       Demag_Apply(DemagData *d, const double *y_dev, double *h_out_dev);

void       Demag_Destroy(DemagData *d);

/* Get self-coupling values (N_αα(0) × strength) for use in the
 * block-Jacobi preconditioner's local 3x3 Jacobian. */
void       Demag_GetSelfCoupling(DemagData *d,
                                 double *nxx0_scaled,
                                 double *nyy0_scaled,
                                 double *nzz0_scaled);

#ifdef __cplusplus
}
#endif
#endif
