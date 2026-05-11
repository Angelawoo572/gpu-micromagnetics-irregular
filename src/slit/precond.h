#ifndef PRECOND_H
#define PRECOND_H

/*
 * precond.h — 3×3 Block-Jacobi preconditioner for spin-wave slit LLG.
 *
 * Each block approximates (I − γ ∂f/∂m_i) using:
 *   - z-axis anisotropy self-coupling (c_msk={0,0,1})
 *   - demag self-coupling N(0) diagonal
 *   - exchange contributes zero self-coupling
 *
 * ymsk: hole cells → P applied via ymsk multiply inside apply_P_kernel.
 * API identical to i2/i5 precond.
 */

#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PrecondData PrecondData;

PrecondData* Precond_Create(int ng, int ny, int ncell);
void         Precond_Destroy(PrecondData *pd);

int PrecondSetup(sunrealtype t, N_Vector y, N_Vector fy,
                 sunbooleantype jok, sunbooleantype* jcurPtr,
                 sunrealtype gamma, void* user_data);

int PrecondSolve(sunrealtype t, N_Vector y, N_Vector fy,
                 N_Vector r, N_Vector z,
                 sunrealtype gamma, sunrealtype delta,
                 int lr, void* user_data);

#ifdef __cplusplus
}
#endif
#endif
