#ifndef PRECOND_H
#define PRECOND_H

/*
 * precond.h  (v2)
 *
 * 3×3 Block-Diagonal Jacobi Preconditioner — with correct gamma handling.
 *
 * Key fix over v1
 * ---------------
 * v1 returned immediately when jok=SUNTRUE, reusing P^{-1} from the
 * previous call.  This is WRONG: jok=SUNTRUE means J is structurally
 * unchanged (y hasn't moved much), but gamma changes every Newton step,
 * so P = I - gamma*J changes too, and P^{-1} must be recomputed.
 *
 * v2 separates J storage from P^{-1} storage:
 *   d_J   : 9 doubles/cell — the raw analytic Jacobian blocks.
 *            Written only when jok=SUNFALSE (J recomputed from h_eff).
 *   d_Pinv: 9 doubles/cell — (I - gamma*J)^{-1}.
 *            Recomputed from d_J on EVERY psetup call (jok=TRUE or FALSE),
 *            because gamma can change even when J doesn't.
 *
 * Memory: 2 × 9 × ncell × 8 bytes = ~174 MB for 1.28M cells.
 *
 * Effect on GMRES convergence
 * ---------------------------
 * With wrong gamma:  preconditioner approximates (I - gamma_old * J)^{-1}
 *                    applied to a system with (I - gamma_new * J).
 *                    The residual after preconditioning is not small,
 *                    so GMRES needs 5 iterations to converge.
 * With correct gamma: preconditioner is exact for the self-coupling block,
 *                     GMRES should converge in 2-3 iterations.
 *
 * API (unchanged from v1 — drop-in replacement)
 * -----------------------------------------------
 *   PrecondData* Precond_Create(int ng, int ny, int ncell);
 *   void         Precond_Destroy(PrecondData *pd);
 *   int          PrecondSetup(...);   // CVODE psetup callback
 *   int          PrecondSolve(...);   // CVODE psolve callback
 */

#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PrecondData {
    sunrealtype *d_J;      /* device: 9 * ncell — raw Jacobian blocks J_ii     */
    sunrealtype *d_Pinv;   /* device: 9 * ncell — (I - gamma*J)^{-1} blocks    */
    sunrealtype  last_gamma; /* gamma used for current d_Pinv (diagnostic only) */
    int          ng;
    int          ny;
    int          ncell;
} PrecondData;

PrecondData* Precond_Create(int ng, int ny, int ncell);
void         Precond_Destroy(PrecondData *pd);

int PrecondSetup(sunrealtype t,
                 N_Vector y, N_Vector fy,
                 sunbooleantype jok, sunbooleantype *jcurPtr,
                 sunrealtype gamma, void *user_data);

int PrecondSolve(sunrealtype t,
                 N_Vector y, N_Vector fy,
                 N_Vector r, N_Vector z,
                 sunrealtype gamma, sunrealtype delta,
                 int lr, void *user_data);

#ifdef __cplusplus
}
#endif

#endif /* PRECOND_H */
