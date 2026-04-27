# `sim_config.h` — General Input Template for `gpu-micromagnetics-irregular`

This template consolidates **every configurable knob** across the five
solver variants (`2d_i1` ... `2d_i5`) into one place. Use it to set up a
new geometry / initial condition / physics regime without having to
rewrite the solver internals.

Source: <https://github.com/Angelawoo572/gpu-micromagnetics-irregular>

## Files

| File | What it does |
|---|---|
| `sim_config.h` | One central C header. Every constant lives here, wrapped in `#ifndef … #define … #endif` so command-line `-D` overrides win. |
| `params.mk` | Makefile fragment that exposes every knob as a `make`-overridable variable and pipes them all into `NVCC_FLAGS` as `-D` defines. |

## Quick start

1. Drop `sim_config.h` into the variant folder you want to clone, e.g.
   `src/irregular/2d_fft/2d_i2/`.
2. At the very top of `2d_fft.cu` add `#include "sim_config.h"` and
   delete that file's per-variant `#ifndef KRYLOV_DIM …` knob block.
3. Drop `params.mk` next to the variant's `Makefile` and add `include
   params.mk` near the top, deleting the old per-knob `?=` block.
4. Rebuild: `make show-config && make clean && make`.

## What each section in `sim_config.h` controls

| Section | Knobs |
|---|---|
| **0. Discretization** | `NX_VAL`, `NY_VAL`. `ng = NX_VAL/3`, `ncell = ng*ny`, `neq = 3*ncell`. |
| **1. Geometry** | `GEOMETRY_KIND` picks bulk / square hole / ring / polycrystal / custom. Each picks up a different sub-block of knobs. |
| **2. Initial condition** | `IC_KIND` picks uniform / head-on stripes / two-domain / grain-bumps / custom. |
| **3. Physics** | exchange `c_che`, anisotropy `c_chk`/`c_cha` and axis `c_msk`, DMI `c_chb` and dir `c_nsk`, damping `c_alpha`, gyromagnetic `c_chg`, applied field `happ_*`. |
| **4. Demag** | `DEMAG_STRENGTH` (0 disables), `DEMAG_THICK`, optional `DEMAG_WINDOWED` (only for polycrystal `2d_i3`). |
| **5. Solver** | `RTOL_VAL`, `ATOL_VAL`, `T_TOTAL`, `MAX_BDF_ORDER`, `KRYLOV_DIM`, `BLOCK_X`, `BLOCK_Y`, `GS_KIND`. |
| **6. Output** | `ENABLE_OUTPUT`, `EARLY_SAVE_UNTIL/EVERY`, `LATE_SAVE_EVERY`, `WRITE_FINAL_STATE`. |

## Cookbook: reproducing each variant

### Reproduce `2d_i1` — head-on three-stripe domain wall

```
make NX_VAL=1536 NY_VAL=512 \
     GEOMETRY_KIND=0  IC_KIND=1 \
     STRIPE_LEFT_FRAC=0.25 STRIPE_RIGHT_FRAC=0.75 \
     INIT_RANDOM_EPS=0.01 INIT_RANDOM_SEED=12345 \
     PHYS_C_CHK=4.0 PHYS_C_CHE=4.0 PHYS_C_CHA=0.0 \
     ANISO_KIND=0  PHYS_MSK_X=1 PHYS_MSK_Y=0 PHYS_MSK_Z=0 \
     PHYS_C_CHB=0.3  PHYS_NSK_X=1 PHYS_NSK_Y=0 PHYS_NSK_Z=0 \
     DEMAG_STRENGTH=2.0 DEMAG_THICK=1.0
```

### Reproduce `2d_i2` — square hole, uniform IC

```
make GEOMETRY_KIND=1  IC_KIND=0 \
     HOLE_CENTER_X_FRAC=0.5 HOLE_CENTER_Y_FRAC=0.5 HOLE_RADIUS_FRAC_Y=0.05 \
     INIT_MX=1.0 INIT_MY=-0.0175 INIT_MZ=0.0 \
     PHYS_C_CHK=1.0 PHYS_C_CHE=50.0 \
     ANISO_KIND=1  PHYS_MSK_X=1 PHYS_MSK_Y=1 PHYS_MSK_Z=1 \
     DEMAG_STRENGTH=4.0 DEMAG_THICK=1.0
```

### Reproduce `2d_i3` — polycrystal with dead-grain holes

Note: this variant uses per-cell easy axis `d_msk` and per-cell DMI
direction `d_nsk` built on the host from grain seeds. The `PHYS_MSK_*`
and `PHYS_NSK_*` macros become fallback values; the kernel reads from
device tables instead. You also need `DEMAG_WINDOWED=1` to apply the
soft mask `w` to the magnetization before the FFT.

```
make GEOMETRY_KIND=3  IC_KIND=3 \
     NUM_GRAINS=72 DEAD_GRAIN_FRAC=0.16 \
     HOLE_SEED=20251104 MASK_EPS_CELLS=2.2 \
     GRAIN_Z_BIAS=1.6 IC_CORE_MZ=0.95 \
     PHYS_C_CHE=4.0 PHYS_C_CHK=1.0 \
     ANISO_KIND=0 PHYS_C_CHB=0.6 \
     DEMAG_STRENGTH=1.0 DEMAG_WINDOWED=1
```

### Reproduce `2d_i4` — square hole + applied field + two-domain IC

```
make GEOMETRY_KIND=1  IC_KIND=2 \
     HOLE_CENTER_X_FRAC=0.5 HOLE_CENTER_Y_FRAC=0.5 HOLE_RADIUS_FRAC_Y=0.05 \
     TWO_DOMAIN_SPLIT_FRAC=0.875 TWO_DOMAIN_TAIL_MY=0.0175 \
     PHYS_C_CHK=1.0 PHYS_C_CHE=20.0 \
     ANISO_KIND=1 PHYS_MSK_X=1 PHYS_MSK_Y=1 PHYS_MSK_Z=1 \
     HAPP_ENABLE=1 HAPP_X=-0.2 HAPP_Y=0 HAPP_Z=0 \
     DEMAG_STRENGTH=4.0 DEMAG_THICK=1.0
```

### Reproduce `2d_i5` — ring (outer rect minus inner rect)

```
make GEOMETRY_KIND=2  IC_KIND=0 \
     OUTER_W_FRAC=0.5 OUTER_H_FRAC=0.5 \
     INNER_W_FRAC_OF_OUTER=0.5 INNER_H_FRAC_OF_OUTER=0.5 \
     PHYS_C_CHK=1.0 PHYS_C_CHE=6.0 \
     ANISO_KIND=1 PHYS_MSK_X=1 PHYS_MSK_Y=1 PHYS_MSK_Z=1 \
     DEMAG_STRENGTH=4.0 DEMAG_THICK=1.0
```

## Adding a new geometry (the recommended way)

The architecture's whole geometry encoding lives in **one** SoA mask
`d_ymsk[3*ncell]` (1 = active, 0 = hole). Every kernel multiplies its
output by this mask, so adding a new shape is purely a host-side mask
build.

1. Set `GEOMETRY_KIND = GEOMETRY_CUSTOM` (= 4).
2. In `main()` of `2d_fft.cu`, allocate the host-side `h_ymsk`, fill it
   with your shape (1 = active, 0 = hole), upload to `d_ymsk` with
   `cudaMemcpy(...)`. That's it — no other kernel changes required.
3. Hole cells have `m=0` and stay frozen by the mask. Hole-neighbor
   reads return 0 automatically, so exchange / DMI / demag sums at the
   active boundary are correct without any branching.

## Things that look like knobs but are NOT

These you'll have to edit in source if you really need them:

| What | Where | Why it's not a macro |
|---|---|---|
| Periodic vs Dirichlet BCs | `wrap_x` / `wrap_y` in 2d_fft.cu, jtv.cu, precond.cu | Affects every kernel; not a one-liner toggle. |
| The byte layout of `UserData` | 2d_fft.cu, jtv.cu (`JtvUserData`), precond.cu (`PcUserData`) | Three structs must stay byte-for-byte mirrors. Extending requires a synchronized change in all three. |
| Material constants in `jtv.cu` and `precond.cu` (`jc_*`, `pc_*`) | Top of those files | They are `__constant__` device variables read by the kernels. Changing physics in `sim_config.h` requires updating these mirrors. |

A clean follow-up refactor is to have `jtv.cu` and `precond.cu` also
`#include "sim_config.h"` and use `PHYS_C_CHK` etc. directly in the
`__constant__` declarations. That eliminates the mirror-maintenance
problem entirely.

## Sanity checks built in

`sim_config.h` `#error`s out at compile time if:

- `NX_VAL` is not a multiple of `GROUPSIZE` (= 3),
- `BLOCK_X * BLOCK_Y > 1024` (CUDA per-block thread limit),
- `GEOMETRY_KIND == GEOMETRY_POLYCRYSTAL` and `NUM_GRAINS < 1`,
- `GEOMETRY_KIND == GEOMETRY_RING` and inner ≥ outer fractions.

`make show-config` prints the full configuration before building.
