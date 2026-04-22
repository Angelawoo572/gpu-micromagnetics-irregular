#ifndef FFT_DEMAG_H
#define FFT_DEMAG_H

/*
 * demag_fft.h  —  FFT demagnetization field using Newell tensor
 *
 * Real-space demag tensor N_αβ(r) is computed with the closed-form
 * Newell/Donahue analytic integrals (calt + ctt), then FFT'd once at
 * startup.  Every RHS call uses:
 *
 *   M̂_β(k)  = FFT[ M_β(r) ]
 *   Ĥ_α(k)  = Σ_β  N̂_αβ(k) · M̂_β(k)
 *   h_α(r)  = IFFT[ Ĥ_α(k) ] / (nx*ny)
 *
 * thick : cell thickness in z, in units of cell spacing
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DemagData DemagData;

DemagData* Demag_Init(int nx, int ny, double thick, double demag_strength);
void       Demag_Apply(DemagData *d, const double *y_dev, double *h_out);
void       Demag_Destroy(DemagData *d);

#ifdef __cplusplus
}
#endif
#endif
