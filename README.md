# GPU Micromagnetic Simulation with Irregular Geometry

## Overview

This project studies GPU-accelerated micromagnetic simulation for the
Landau-Lifshitz-Gilbert (LLG) equation using CUDA and SUNDIALS/CVODE.

The code supports:

- local nearest-neighbor interactions,
- 2D magnetization dynamics,
- irregular masked geometries,
- FFT-based long-range demagnetization,
- and solver-level performance analysis.

The main goal is to understand end-to-end GPU performance, including not only
the physics kernels, but also the CVODE solver pipeline, vector operations,
synchronization, and FFT-based global coupling.

Project page:

```text
https://angelawoo572.github.io/gpu-micromagnetics-irregular/
```

---

## Repository Structure

```text
gpu-micromagnetics-irregular/
├── src/
├── configs/
├── results/
├── report/
├── Makefile
└── README.md
```

### `src/`

All simulation source code is stored here.

Important subfolders:

```text
src/1d/
```

Early 1D local-interaction prototype used for correctness checks.

```text
src/2d/
```

2D local-interaction solver without FFT demagnetization.

```text
src/2d_fft/
```

2D solver with FFT-based demagnetization.

```text
src/irregular/
```

Irregular-geometry solvers and experiments, including masked geometries,
square holes, ring/active-region cases, polycrystal cases, and compact
active-cell execution.

Typical variants include:

```text
2d_i1  head-on stripe / local or FFT baseline
2d_i2  square-hole geometry with uniform initialization
2d_i3  Voronoi polycrystal + dead-grain holes + FFT demag
2d_i4  square-hole geometry with asymmetric two-domain initialization
2d_i5  ring / central active-region style geometry
2d_i6  compact active-cell geometry such as ellipse/circle
```

---

## `configs/`

This folder stores input/configuration templates for running simulations.

The main files are:

```text
configs/sim_config.h
configs/params.mk
```

### `sim_config.h`

Central C/CUDA configuration header.

It contains the main simulation knobs:

- grid size: `NX_VAL`, `NY_VAL`
- execution model: dense mask or compact active-cell execution
- geometry type: bulk, square hole, ring, polycrystal, ellipse, custom
- initial condition type
- physical constants
- demag strength and thickness
- solver tolerance and Krylov dimension
- output schedule

### `params.mk`

Makefile fragment that exposes the same knobs as command-line variables.

Example:

```bash
make NX_VAL=1536 NY_VAL=512 DEMAG_STRENGTH=2.0 RTOL_VAL=1.0e-4
```

Use `configs/` when you want to reproduce or modify a case without rewriting
the CUDA source code.

---

## `results/`

This folder contains generated outputs and analysis results.

Look here for:

- correctness figures,
- magnetization visualizations,
- timing results,
- scaling plots,
- Nsight profiling summaries,
- direct-vs-FFT comparisons,
- irregular-geometry result images.

The results are meant to show both physical behavior and performance behavior.

---

## `report/`

This folder contains report materials, figures, and final writeups.

The report explains the main design choices:

- regular-grid baseline,
- irregular masked geometry,
- dense mask vs compact active-cell execution,
- solver-level optimizations,
- FFT demagnetization,
- profiling interpretation.

Use this folder if you want the full explanation behind the code and results.

---

## Main Simulation Ideas

### 1. Local GPU RHS

The local part of the LLG effective field is computed with CUDA kernels.
Each cell stores three magnetization components in structure-of-arrays format:

```text
[mx for all cells][my for all cells][mz for all cells]
```

This layout matches the SUNDIALS CUDA NVector interface and works well with
grid-based stencil computation.

### 2. Irregular Geometry

Irregular geometries are embedded inside a structured grid.

Two execution models are used:

```text
EXEC_YMSK
```

Dense masked execution. Kernels launch over the full grid, and inactive cells
are masked out.

```text
EXEC_COMPACT
```

Compact active-cell execution. Kernels launch only over active cells using
active-cell index lists.

Dense masking is simpler. Compact execution is useful when many cells are
inactive.

### 3. FFT Demagnetization

The long-range demagnetization field is computed using FFT convolution.

The direct method is expensive:

```text
O(N^2)
```

The FFT method reduces this to approximately:

```text
O(N log N)
```

The FFT pipeline is GPU-resident:

```text
gather magnetization
→ cuFFT D2Z
→ spectral tensor multiply
→ cuFFT Z2D
→ scatter demag field
```

The demag tensor is computed once during initialization and stored on the GPU.

### 4. Solver-Level Optimization

The project also studies CVODE/SPGMR overhead.

Important solver-level components include:

- fused SUNDIALS NVector operations,
- multi-dot reduction,
- analytic Jacobian-vector product,
- block-Jacobi preconditioner,
- CUDA stream organization.

Profiling shows that total runtime is often dominated by solver orchestration,
vector kernels, reductions, and synchronization, not only by the physics RHS
kernel.

---

## Build

A top-level `Makefile` is provided.

Basic commands:

```bash
make
make run
make clean
```

To show the active configuration:

```bash
make show-config
```

To override parameters from the command line:

```bash
make NX_VAL=1536 NY_VAL=512 RTOL_VAL=1.0e-4 DEMAG_STRENGTH=2.0
```

---

## Example Runs

### Square-hole case

```bash
make EXECUTION_MODEL=0 \
     GEOMETRY_KIND=1 \
     IC_KIND=0 \
     DEMAG_STRENGTH=4.0
```

### Polycrystal FFT case

```bash
make EXECUTION_MODEL=0 \
     GEOMETRY_KIND=3 \
     IC_KIND=3 \
     NUM_GRAINS=72 \
     DEAD_GRAIN_FRAC=0.16 \
     DEMAG_STRENGTH=1.0 \
     DEMAG_WINDOWED=1
```

### Compact ellipse / circle case

```bash
make EXECUTION_MODEL=1 \
     GEOMETRY_KIND=5 \
     ACTIVE_RX_FRAC=0.25 \
     ACTIVE_RY_FRAC=0.25 \
     BLOCK_SIZE=256
```

---

## Output

Depending on the selected configuration, the simulation may write magnetization
states or final-state data.

Common output contains:

```text
time nx ny
mx my mz
mx my mz
...
```

These files can be postprocessed into:

- component maps such as `mx`, `my`, `mz`,
- vector-field plots,
- 3D surface plots,
- GIF animations,
- timing and scaling plots.

Generated figures and processed results should be stored in `results/`.

---

## Validation

Correctness is checked in stages:

1. 1D local-interaction behavior.
2. Representative magnetization-vector trajectories.
3. 2D local-only vector fields.
4. Small FFT-demag cases compared against direct convolution where possible.
5. Irregular-geometry cases checked through physical consistency and visual
   evolution.

---

## Performance Evaluation

The project evaluates:

- total runtime / time-to-solution,
- scaling with grid size,
- dense mask vs compact active-cell execution,
- direct demag vs FFT demag,
- solver tolerance effects,
- CUDA kernel breakdown,
- CUDA API synchronization overhead,
- SUNDIALS vector-operation overhead.

Nsight Systems is used to identify where time is spent across CUDA kernels,
cuFFT calls, SUNDIALS vector kernels, and synchronization.

---

## Notes for New Users

Start here:

```text
src/
```

for the actual CUDA/CVODE solvers.

Use:

```text
configs/
```

to understand or modify simulation inputs.

Check:

```text
results/
```

to view generated plots, timing data, and example outputs.

Read:

```text
report/
```

for the full explanation of the method, experiments, and performance analysis.

---

## Conclusion

This project demonstrates that end-to-end performance of GPU-based
micromagnetic simulation is determined not by individual kernels,
but by the interaction between data-parallel computation and
solver-level orchestration.

While local kernels and FFT demagnetization can be highly optimized,
the dominant cost shifts to SUNDIALS vector operations and
synchronization overhead.

Future improvements require changes at the solver level,
such as reducing kernel launch overhead or restructuring
integration pipelines, rather than further tuning physics kernels.