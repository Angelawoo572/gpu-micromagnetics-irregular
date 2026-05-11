#ifndef PRECOND_H
#define PRECOND_H

/*
 * precond.h — Block-Jacobi preconditioner for the Maxwell BDF/Newton system.
 *
 * The Maxwell RHS is a pure curl operator: a 3×3 block per cell with NO
 * self-coupling (∂f_α/∂y_α |_self = 0).  So the local 3×3 J is the zero
 * matrix and (I − γ J) = I.  In other words the trivial block-Jacobi
 * preconditioner with all blocks = identity is exact for the local part.
 *
 * We still expose the full Setup/Solve interface so the SUNDIALS plumbing
 * is identical to the LLG code; PrecondSolve simply copies r → z and
 * zeros it inside the screen.  The off-diagonal coupling (the curl) is
 * left to GMRES, which converges fast because the system is small and
 * well-conditioned.
 */

#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PrecondData PrecondData;

PrecondData* Precond_Create(int nx, int ny, int ncell);
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
