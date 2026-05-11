#ifndef JTV_H
#define JTV_H

/*
 * jtv.h — Analytic Jacobian-vector product for the 2D Maxwell RHS.
 *
 * The Maxwell RHS is LINEAR in the state (Ez, Hx, Hy), so the analytic
 * Jacobian = the linear operator itself.  Jv is therefore the same
 * stencil as f() but with the source term dropped (the source doesn't
 * depend on y).
 *
 * Signature matches SUNDIALS CVLsJacTimesVecFn.
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
