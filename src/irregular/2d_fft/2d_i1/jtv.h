#ifndef JTV_H
#define JTV_H

/*
 * jtv.h — Analytic Jacobian-vector product for the 2D LLG RHS.
 *
 * Signature matches SUNDIALS CVLsJacTimesVecFn. Registered through
 *     CVodeSetJacTimes(cvode_mem, NULL, JtvProduct);
 *
 * The analytic Jv covers exchange + x-axis anisotropy + DMI (all linear
 * in y, local stencil). Demag contribution to Jv is NOT included:
 * adding it would require a second FFT pipeline per Jv call, and GMRES
 * plus the preconditioner tolerate the inexactness without issue.
 */

#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

int JtvProduct(N_Vector v, N_Vector Jv, sunrealtype t,
               N_Vector y, N_Vector fy, void* user_data, N_Vector tmp);

#ifdef __cplusplus
}
#endif
#endif
