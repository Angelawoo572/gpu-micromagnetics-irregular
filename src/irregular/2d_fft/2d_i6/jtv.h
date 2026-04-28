#ifndef JTV_H
#define JTV_H

/*
 * jtv.h — Analytic Jacobian-vector product for the 2D LLG RHS.
 *
 * Signature matches SUNDIALS CVLsJacTimesVecFn. Registered through
 *     CVodeSetJacTimes(cvode_mem, NULL, JtvProduct);
 *
 * The analytic Jv covers exchange + per-component Landau anisotropy
 * (all linear in y, local stencil except for the cubic anisotropy term
 * which is handled via its derivative). Demag contribution to Jv is
 * NOT included: adding it would require a second FFT pipeline per Jv
 * call, and GMRES plus the preconditioner tolerate the inexactness
 * without issue.
 *
 * i6 variant: the kernel runs over a compact list of active cell
 * indices; hole-cell entries of Jv are zeroed by a small dedicated
 * kernel inside JtvProduct.
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
