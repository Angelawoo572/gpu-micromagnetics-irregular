好，这个是**整理后的完整版 README（已经帮你压到“用户导航优先 + 不啰嗦 + 结构清楚”版本）**，可以直接用：

---

````markdown
# GPU Micromagnetic Simulation with Irregular Geometry

## Overview

This project implements a GPU-accelerated micromagnetic simulation pipeline
for the Landau–Lifshitz–Gilbert (LLG) equation.

The system combines:

- CUDA kernels for local interactions
- cuFFT for long-range demagnetization
- SUNDIALS/CVODE for time integration
- irregular geometry support (masked + compact execution)

The focus of the project is **end-to-end performance**, not just kernel speed:
we study how solver overhead, synchronization, FFT, and geometry interact.

---

## Repository Structure

```text
gpu-micromagnetics-irregular/
├── src/        # all simulation code
├── configs/    # input/config templates
├── results/    # output, plots, profiling results
├── report/     # report, poster, figures
├── Makefile
└── README.md
````

---

## Where to Find Things

### `src/` — all code

All implementations live here.

```text
src/
├── 1d/           # 1D prototype (correctness)
├── 2d/           # 2D local-only solver
├── 2d_fft/       # 2D + FFT demag
└── irregular/    # irregular geometry + final variants
```

Main final experiments:

```text
src/irregular/2d_fft/2d_i1/
src/irregular/2d_fft/2d_i2/
src/irregular/2d_fft/2d_i3/
src/irregular/2d_fft/2d_i4/
src/irregular/2d_fft/2d_i5/
src/irregular/2d_fft/2d_i6/
```

Each `2d_i*` = one **experiment variant** (geometry + IC + execution model).

---

### `configs/` — input files (IMPORTANT)

This is where you control the simulation **without touching code**.

```text
configs/
├── sim_config.h   # central config (all knobs)
└── params.mk      # Makefile overrides
```

Use these to set:

* grid size (`NX_VAL`, `NY_VAL`)
* geometry (`GEOMETRY_KIND`)
* initial condition (`IC_KIND`)
* physics constants
* demag strength
* solver tolerance
* execution model (masked vs compact)

Example:

```bash
make NX_VAL=1536 NY_VAL=512 DEMAG_STRENGTH=2.0 RTOL_VAL=1e-4
```

---

### `results/` — outputs you can directly view

Contains:

* simulation outputs
* magnetization plots
* timing results
* Nsight profiling summaries
* figures used in report/poster

👉 If you just want to *see results*, start here.

---

### `report/` — writeups

Contains:

* final report
* poster
* figures

Project summary and performance analysis are based on this:


---

## Main Variants (i1–i6)

| Variant | Description                          |
| ------- | ------------------------------------ |
| i1      | head-on stripe/domain wall           |
| i2      | square hole                          |
| i3      | polycrystal (Voronoi + dead grains)  |
| i4      | hole + applied field                 |
| i5      | ring geometry                        |
| i6      | compact active-cell (ellipse/circle) |

Key difference:

* **i1–i5** → masked full-grid execution
* **i6** → compact active-cell execution (faster for sparse domains)

---

## Build & Run

### Basic

```bash
make
make run
```

### Run a specific variant

```bash
cd src/irregular/2d_fft/2d_i6
make
make run
```

### Show configuration

```bash
make show-config
```

---

## Recommended Workflow

1. Pick a variant in `src/`
2. Configure parameters via `configs/` or command line
3. Build
4. Run
5. Check results in `results/`

Example:

```bash
cd src/irregular/2d_fft/2d_i6

make show-config
make clean
make

make EXECUTION_MODEL=1 \
     GEOMETRY_KIND=5 \
     ACTIVE_RX_FRAC=0.25 \
     ACTIVE_RY_FRAC=0.25 \
     DEMAG_STRENGTH=4.0 \
     RTOL_VAL=1e-4

make run
```

---

## Key Configuration Knobs

| Category  | Knobs                              |
| --------- | ---------------------------------- |
| Grid      | `NX_VAL`, `NY_VAL`                 |
| Geometry  | `GEOMETRY_KIND`                    |
| IC        | `IC_KIND`                          |
| Demag     | `DEMAG_STRENGTH`                   |
| Solver    | `RTOL_VAL`, `ATOL_VAL`, `T_TOTAL`  |
| Execution | `EXECUTION_MODEL`                  |
| CUDA      | `BLOCK_X`, `BLOCK_Y`, `BLOCK_SIZE` |
| Output    | `ENABLE_OUTPUT`                    |

---

## Execution Models

### 1. Masked (default, i1–i5)

* full grid launch
* inactive cells masked (`ymsk`)
* simpler, but wastes work

### 2. Compact (i6)

* only launch active cells
* uses index lists
* better when geometry is sparse

---

## Example Runs

### Square hole

```bash
make EXECUTION_MODEL=0 \
     GEOMETRY_KIND=1 \
     IC_KIND=0 \
     HOLE_RADIUS_FRAC_Y=0.05 \
     DEMAG_STRENGTH=4.0
```

### Polycrystal

```bash
make EXECUTION_MODEL=0 \
     GEOMETRY_KIND=3 \
     NUM_GRAINS=72 \
     DEAD_GRAIN_FRAC=0.16 \
     DEMAG_STRENGTH=1.0
```

### Ring

```bash
make EXECUTION_MODEL=0 \
     GEOMETRY_KIND=2 \
     OUTER_W_FRAC=0.5 \
     INNER_W_FRAC_OF_OUTER=0.5
```

### Compact ellipse (recommended)

```bash
make EXECUTION_MODEL=1 \
     GEOMETRY_KIND=5 \
     ACTIVE_RX_FRAC=0.25 \
     ACTIVE_RY_FRAC=0.25 \
     BLOCK_SIZE=256
```

---

## Outputs & Metrics

Typical outputs:

* magnetization field
* final state
* runtime
* CVODE stats

Important metrics:

| Metric | Meaning              |
| ------ | -------------------- |
| `nst`  | time steps           |
| `nfe`  | RHS calls            |
| `nni`  | nonlinear iterations |
| `nli`  | linear iterations    |
| `ncfn` | nonlinear failures   |
| `netf` | error test failures  |

---

## Performance Insight (What this project shows)

Key takeaway:

> The bottleneck is NOT just the physics kernel.

From experiments and profiling:

* solver overhead dominates
* synchronization (`cudaStreamSynchronize`) is major cost
* many small vector kernels (SUNDIALS) dominate runtime
* FFT adds cost but is not the main bottleneck
* compact execution gives modest speedup unless sparsity is high

This means optimization must target:

* solver pipeline
* vector operations
* synchronization
* not just CUDA kernels

---

## Dependencies

* CUDA
* cuFFT
* SUNDIALS (with CUDA)
* Nsight Systems (profiling)

---

## Links

Project page:

```
https://angelawoo572.github.io/gpu-micromagnetics-irregular/
```

GitHub:

```
https://github.com/Angelawoo572/gpu-micromagnetics-irregular
```

---

## TL;DR

* Code → `src/`
* Inputs → `configs/`
* Outputs → `results/`
* Analysis → `report/`

Run a variant, tweak configs, check results.

```