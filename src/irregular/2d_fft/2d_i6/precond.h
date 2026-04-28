#ifndef PRECOND_H
#define PRECOND_H

/*
 * precond.h — Block-Jacobi preconditioner for the LLG BDF/Newton system.
 *
 * Each 3×3 diagonal block P_i approximates  (I − γ ∂f/∂m_i),  where ∂f/∂m_i
 * is built from the LOCAL part of the effective field only:
 *   exchange self-term (vanishes for 5-point stencil at the center cell),
 *   per-component Landau anisotropy chk·m_α(m_α² − 1)
 *       → diagonal contribution chk·(3 m_α² − 1),
 *   demag self-coupling N(0) (diagonal by 4-fold symmetry).
 *
 * The 3×3 block is inverted explicitly on device and stored in a
 * full-grid array (size 9*ncell, indexed by cell).  In the i6 variant
 * only ACTIVE cells are built, and PrecondSolve only applies P⁻¹ at
 * active cells; hole-cell entries of z are zeroed by a small kernel.
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
