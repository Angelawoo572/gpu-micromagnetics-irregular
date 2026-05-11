#ifndef JTV_H
#define JTV_H

/*
 * jtv.h — Analytic Jacobian-vector product for the spin-wave slit LLG RHS.
 *
 * Covers: exchange + z-axis anisotropy + DMI (local stencil, linear in y).
 * Demag Jv excluded (would require second FFT per GMRES iteration).
 * Hole cells masked by ymsk → Jv = 0 there.
 *
 * Registered via: CVodeSetJacTimes(cvode_mem, NULL, JtvProduct)
 */

#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

int JtvProduct(N_Vector v, N_Vector Jv, sunrealtype t,
               N_Vector y, N_Vector fy,
               void *user_data, N_Vector tmp);

#ifdef __cplusplus
}
#endif
#endif
