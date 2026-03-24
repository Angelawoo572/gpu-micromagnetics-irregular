# GPU Micromagnetic Simulation with Irregular Geometry

## Overview

This project studies GPU-accelerated micromagnetic simulation for the
Landau-Lifshitz-Gilbert (LLG) equation. The goal is to build a CUDA-based
simulation pipeline that supports:

- local nearest-neighbor interactions,
- 2D magnetization dynamics,
- FFT-based long-range magnetostatic computation,
- and irregular masked geometries on structured grids.

The project focuses not only on implementing a working solver, but also on
understanding how data layout and geometry affect performance on GPUs.

## Main Research Questions

This project is organized around the following systems questions:

1. How should a GPU-based micromagnetic solver organize local computation,
   long-range FFT computation, and time integration efficiently?
2. How does irregular geometry affect performance compared with a regular grid?
3. What tradeoffs arise between dense masked execution and compacted active-cell execution?
4. How do data layout choices influence local kernel efficiency and FFT efficiency?

## Current Project Structure

- `src/1d/`  
  Early 1D local-interaction prototype used for correctness checking.

- `src/2d/`  
  2D local-interaction solver without FFT-based long-range coupling.

- `src/2d_fft/`  
  Main 2D solver with FFT-based magnetostatics.

- `src/irregular/`  
  Utilities and experiments related to masked irregular geometries and active-cell execution.

- `report/`  
  Proposal source and figures. Project webpage materials and PDFs.

- `results/`  
  Correctness figures, timing results, and performance plots.

## Correctness Validation

Current validation is staged:

- **1D local-interaction results**  
  Used to verify qualitative spatial magnetization evolution.

- **Representative 3D magnetization traces**  
  Used to verify plausible LLG dynamics for individual vectors.

- **2D local-only vector fields**  
  Used to validate the 2D extension before integrating FFT-based long-range terms.

The final system will additionally validate small FFT-based cases against
simplified or direct reference computations where possible.

## Planned Evaluation

We will evaluate both correctness and performance.

### Correctness
- 1D magnetization evolution
- representative vector trajectories
- 2D local-only field structure
- consistency checks for small FFT-based cases

### Performance
- total runtime / time-to-solution
- runtime breakdown:
  - local RHS / field computation
  - FFT-based long-range computation
  - solver overhead
- scaling with grid size
- impact of irregular geometry
- comparison of alternative layout / execution strategies

## Build

A top-level `Makefile` is provided.

Example targets:

```bash
make run
make clean