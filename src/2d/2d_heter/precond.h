#ifndef PRECOND_H
#define PRECOND_H

/*
 * precond.h
 *
 * 3×3 Block-Diagonal Jacobi Preconditioner for the 2D LLG solver.
 *
 * Motivation (large-problem regime)
 * ----------------------------------
 * For large problems (3000×1280, neq=3.84M) the bottleneck is not
 * sync count but the total volume of GMRES vector operations:
 *   linearSumKernel, scaleKernel, dotProdKernel, divKernel, ...
 * Each Newton step calls SPGMR which by default runs up to 5 Krylov
 * iterations; each iteration does ~10 vector ops.  The preconditioner
 * makes GMRES converge in 1-2 iterations instead, cutting the vector
 * op count by 3-5x and achieving the speedup that fused-NVector alone
 * cannot deliver for large n.
 *
 * Preconditioner construction
 * ----------------------------
 * For cell i, ignore inter-cell coupling (treat neighbors as frozen).
 * The local effective field h(m_i) has only one self-dependent term:
 *   h3 += c_msk[2] * c_chk * m3   (easy-axis anisotropy in z)
 * This gives a 3×3 analytic Jacobian J_ii = ∂f_i / ∂m_i.
 *
 * CVODE BDF needs to solve  (I - γ J) δ = r  at each Newton step.
 * We build P = I - γ J_block_diag and store P^{-1} via Cramer's rule.
 * psolve then applies z = P^{-1} r with one CUDA kernel (one thread/cell).
 *
 * Storage: 9 sunrealtype per cell → ~92 MB for 1.28M cells (fine).
 *
 * API
 * ---
 *   PrecondData* Precond_Create(int ng, int ny, int ncell, SUNContext sunctx);
 *   void         Precond_Destroy(PrecondData* pd);
 *
 *   // CVODE-compatible callbacks (pass as CVodeSetPreconditioner args):
 *   int PrecondSetup(sunrealtype t, N_Vector y, N_Vector fy,
 *                    sunbooleantype jok, sunbooleantype *jcurPtr,
 *                    sunrealtype gamma, void *user_data);
 *   int PrecondSolve(sunrealtype t, N_Vector y, N_Vector fy,
 *                    N_Vector r, N_Vector z,
 *                    sunrealtype gamma, sunrealtype delta,
 *                    int lr, void *user_data);
 *
 * The UserData struct must contain a `PrecondData *pd` field.
 * PrecondSetup/PrecondSolve access it through user_data.
 */

#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PrecondData {
    sunrealtype *d_Pinv;   /* device: 9 * ncell doubles — per-cell P^{-1} */
    int          ng;
    int          ny;
    int          ncell;
} PrecondData;

/* Allocate Pinv storage on device */
PrecondData* Precond_Create(int ng, int ny, int ncell);

/* Free device storage */
void Precond_Destroy(PrecondData *pd);

/*
 * CVODE psetup callback.
 *
 * Called whenever CVODE decides the preconditioner needs updating
 * (jok=SUNFALSE) or can be reused (jok=SUNTRUE).
 * Computes P = I - gamma * J_block_diag and stores P^{-1} per cell.
 *
 * user_data must point to a struct whose FIRST member is PrecondData*.
 * (See UserDataFull in 2d_p.cu)
 */
int PrecondSetup(sunrealtype t,
                 N_Vector y, N_Vector fy,
                 sunbooleantype jok, sunbooleantype *jcurPtr,
                 sunrealtype gamma, void *user_data);

/*
 * CVODE psolve callback.
 *
 * Applies z = P^{-1} r using stored Pinv blocks.
 * lr = 1 (left) or 2 (right); we implement left preconditioning.
 */
int PrecondSolve(sunrealtype t,
                 N_Vector y, N_Vector fy,
                 N_Vector r, N_Vector z,
                 sunrealtype gamma, sunrealtype delta,
                 int lr, void *user_data);

#ifdef __cplusplus
}
#endif

#endif /* PRECOND_H */
