/*
 * precond.cu — Trivial block-Jacobi preconditioner for Maxwell.
 *
 * For a pure curl RHS, ∂f_α/∂y_α |_self = 0 at every cell, so
 *   A_local = I − γ · 0 = I,  and  P⁻¹ = I.
 *
 * PrecondSolve therefore just copies r → z and zeros the result inside
 * the PEC screen.  This mirrors the LLG code's API exactly, so
 * CVODE/SPGMR sees the same plumbing.
 */

#include "precond.h"

#include <cvode/cvode.h>
#include <nvector/nvector_cuda.h>
#include <sundials/sundials_types.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#ifndef PC_BLOCK_SIZE
#define PC_BLOCK_SIZE 256
#endif

struct PrecondData {
    int nx, ny, ncell;
};

typedef struct {
    void        *pd_opaque;
    int         *d_active_ids;
    int         *d_inactive_ids;
    int nx, ny, ncell, neq;
    int n_active, n_inactive;
    int src_col, pad0;
    double inv_dx;
    double omega;
    double t_ramp;
} PcUserData;

__global__ static void pc_copy_kernel(
    const sunrealtype* __restrict__ r,
    sunrealtype*       __restrict__ z,
    int total_len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_len) return;
    z[tid] = r[tid];
}

__global__ static void pc_zero_inactive_kernel(
    sunrealtype* __restrict__ z,
    const int* __restrict__ inactive_ids,
    int n_inactive,
    int ncell)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_inactive) return;
    int cell = inactive_ids[tid];
    z[cell]           = SUN_RCONST(0.0);
    z[ncell + cell]   = SUN_RCONST(0.0);
    z[2*ncell + cell] = SUN_RCONST(0.0);
}

PrecondData* Precond_Create(int nx, int ny, int ncell)
{
    PrecondData *pd = (PrecondData*)calloc(1, sizeof(PrecondData));
    if (!pd) { fprintf(stderr, "[Precond] calloc failed\n"); return NULL; }
    pd->nx = nx; pd->ny = ny; pd->ncell = ncell;
    printf("[Precond] Trivial identity block-Jacobi (Maxwell): ncell=%d\n",
           ncell);
    return pd;
}

void Precond_Destroy(PrecondData *pd) { if (pd) free(pd); }

int PrecondSetup(sunrealtype t, N_Vector y, N_Vector fy,
                 sunbooleantype jok, sunbooleantype* jcurPtr,
                 sunrealtype gamma, void* user_data)
{
    (void)t; (void)y; (void)fy; (void)jok; (void)gamma; (void)user_data;
    *jcurPtr = SUNTRUE;
    return 0;
}

int PrecondSolve(sunrealtype t, N_Vector y, N_Vector fy,
                 N_Vector r, N_Vector z,
                 sunrealtype gamma, sunrealtype delta,
                 int lr, void* user_data)
{
    (void)t; (void)y; (void)fy; (void)gamma; (void)delta; (void)lr;
    PcUserData *ud = (PcUserData*)user_data;

    sunrealtype *rdata = N_VGetDeviceArrayPointer_Cuda(r);
    sunrealtype *zdata = N_VGetDeviceArrayPointer_Cuda(z);

    /* z = r */
    int total = 3 * ud->ncell;
    int g = (total + PC_BLOCK_SIZE - 1) / PC_BLOCK_SIZE;
    pc_copy_kernel<<<g, PC_BLOCK_SIZE>>>(rdata, zdata, total);

    /* zero hole entries of z */
    if (ud->n_inactive > 0) {
        int g0 = (ud->n_inactive + PC_BLOCK_SIZE - 1) / PC_BLOCK_SIZE;
        pc_zero_inactive_kernel<<<g0, PC_BLOCK_SIZE>>>(
            zdata, ud->d_inactive_ids, ud->n_inactive, ud->ncell);
    }

    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "[PrecondSolve] failed: %s\n",
                cudaGetErrorString(cudaGetLastError()));
        return -1;
    }
    return 0;
}
