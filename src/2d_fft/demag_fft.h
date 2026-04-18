#ifndef FFT_DEMAG_H
#define FFT_DEMAG_H

/*
 * fft_demag.h  —  FFT-based demagnetization field (cuFFT, double precision)
 *
 * Implements the convolution theorem for the dipolar demag field:
 *
 *   h_dmag(i,j) = Σ_{m,n} D(i-m, j-n) · M(m,n)     [space domain]
 *               = IFFT[ D̂(kx,ky) · M̂(kx,ky) ]      [k-space, O(N log N)]
 *
 * Usage:
 *   // Once at startup:
 *   DemagData *demag = Demag_Init(nx, ny, demag_strength);
 *
 *   // Inside RHS f():
 *   Demag_Apply(demag, y_device_ptr, h_field_device_ptr);
 *
 *   // At shutdown:
 *   Demag_Destroy(demag);
 *
 * The h_field pointer must point to an already-zeroed (or exchange-filled)
 * SoA buffer [hx_0..hx_{N-1}][hy_0..hy_{N-1}][hz_0..hz_{N-1}].
 * Demag_Apply ADDS h_dmag to it.
 *
 * Memory: ~7 × (ny × (nx/2+1)) × 16 bytes of complex doubles on device.
 * For nx=1000, ny=1280: ~7 × 640640 × 16 ≈ 72 MB.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct DemagData DemagData;

/*
 * Demag_Init
 *
 * Allocates cuFFT plans, device buffers, and precomputes D̂(k).
 *
 * nx, ny          : physical grid dimensions (columns × rows)
 * demag_strength  : scaling factor for demag field (1.0 = full physics;
 *                   0.0 = disabled; use small values to ramp up)
 *
 * Returns: pointer to DemagData on success, NULL on failure.
 */
DemagData* Demag_Init(int nx, int ny, double demag_strength);

/*
 * Demag_Apply
 *
 * Computes and accumulates the demagnetization field into h_out.
 *
 * y_dev  : device pointer to SoA magnetization [mx|my|mz], 3*nx*ny doubles
 * h_out  : device pointer to SoA effective field, 3*nx*ny doubles
 *          (Demag_Apply ADDS to h_out — caller must initialize it first)
 *
 * Async on stream 0.  No explicit sync issued.
 */
void Demag_Apply(DemagData *d,
                 const double *y_dev,
                 double       *h_out);

/*
 * Demag_Destroy
 *
 * Frees all device memory and cuFFT plans.
 */
void Demag_Destroy(DemagData *d);

#ifdef __cplusplus
}
#endif

#endif /* FFT_DEMAG_H */
