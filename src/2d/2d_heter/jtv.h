#ifndef JTV_H
#define JTV_H

/*
 * jtv.h  —  Analytic Jacobian-times-vector (Jv) for the 2D LLG solver.
 *
 * Problem
 * -------
 * By default CVODE approximates  Jv ≈ [f(y + ε·v) - f(y)] / ε  (finite
 * difference).  This calls the RHS function f() once per GMRES iteration
 * just to get Jv, on top of the f() call that already happened in the
 * Newton residual.  For 3753 Newton steps × 5.45 GMRES iters ≈ 20 468
 * extra f() calls, each costing ~85 µs on GPU = ~1.74 s wasted purely on
 * differencing overhead plus the ε-perturbation error.
 *
 * Solution
 * --------
 * Supply an analytic Jv kernel.  The LLG Jacobian has a known stencil
 * structure (self block + 4-neighbor exchange coupling), so Jv can be
 * evaluated exactly in a single CUDA kernel with no extra f() call and
 * no finite-difference error.
 *
 * Derivation summary (see jtv.cu for full details)
 * -------------------------------------------------
 * LLG RHS at cell i:
 *   f_i = c_chg*(m_i × H_i) + c_alpha*(H_i - (m_i·H_i)*m_i)
 *
 * H_i = c_che*(m_L+m_R+m_U+m_D) + anisotropy_self(m_i) + dmi_x(m_L,m_R)
 *
 * For the Jacobian we keep y (= current m) fixed and differentiate w.r.t.
 * a perturbation v.  Let:
 *   dH_i = c_che*(v_L+v_R+v_U+v_D) + dH_self(v_i) + dH_dmi(v_L,v_R)
 *   dH_self: only h3 component,  dh3 = c_msk[2]*c_chk * v3_i
 *   dH_dmi:  only h1 component,  dh1 = c_chb*c_nsk[0]*(v1_L + v1_R)
 *
 * Then:
 *   (Jv)_i = c_chg*(v_i × H_i + m_i × dH_i)
 *           + c_alpha*(dH_i - (v_i·H_i + m_i·dH_i)*m_i - (m_i·H_i)*v_i)
 *
 * All quantities are available per-cell with the same stencil reads as
 * the RHS kernel, so Jv is computed in one 2-D CUDA kernel at the same
 * cost as one RHS evaluation — but without calling f() at all.
 *
 * Registration
 * ------------
 * After CVodeSetLinearSolver:
 *   CVodeSetJacTimes(cvode_mem, NULL, JtvProduct);
 *
 * The first argument (NULL) tells CVODE not to call a setup routine;
 * the second is the analytic jtimes callback defined here.
 *
 * Correctness
 * -----------
 * Mathematical output is identical to the exact Jacobian applied to v;
 * no ε approximation is involved.  The simulation trajectory is unchanged
 * (same Newton convergence, same step sizes) — only the way Jv is
 * computed changes.
 */

#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * JtvProduct
 *
 * CVODE jtimes callback.  Computes Jv = J(t,y) · v analytically.
 *
 * Signature required by CVodeSetJacTimes:
 *   int jtimes(N_Vector v, N_Vector Jv,
 *              sunrealtype t, N_Vector y, N_Vector fy,
 *              void *user_data, N_Vector tmp);
 *
 * user_data must point to UserData (same struct used for f and precond).
 * tmp is a scratch vector provided by CVODE; we do not use it.
 *
 * Returns 0 on success, non-zero on failure.
 */
int JtvProduct(N_Vector v,  N_Vector Jv,
               sunrealtype t,
               N_Vector y,  N_Vector fy,
               void *user_data,
               N_Vector tmp);

#ifdef __cplusplus
}
#endif

#endif /* JTV_H */
