# `sim_config.h` — General Input Template for `gpu-micromagnetics-irregular`

**v2** — covers `2d_i1` ... `2d_i6`.

This template consolidates **every configurable knob** across the variants
into one place. Use it to set up a new geometry / initial condition /
physics regime without rewriting the solver internals.

Source: <https://github.com/Angelawoo572/gpu-micromagnetics-irregular>

## Files

| File | What it does |
|---|---|
| `sim_config.h` | One central C header. Every constant lives here, wrapped in `#ifndef … #define … #endif` so command-line `-D` overrides win. |
| `params.mk`    | Makefile fragment that exposes every knob as a `make`-overridable variable and pipes them all into `NVCC_FLAGS` as `-D` defines. |

## What's new in v2

`2d_i6` introduced two big changes that the v1 template didn't anticipate:

1. **Two execution models.** `i1..i5` use a byte-mask `d_ymsk` and run every
   kernel over the full grid. `i6` switched to *compact active-cell
   execution*: two index lists `d_active_ids` / `d_inactive_ids`, and every
   per-cell kernel launches with one thread per **active** cell. Hole-cell
   entries of `ydot`/`Jv`/`z` are zeroed by tiny separate kernels so
   SUNDIALS' inner products see consistent zeros.

   The two models cannot mix in one variant — they imply different
   `UserData` byte layouts and different kernel signatures. The new
   `EXECUTION_MODEL` knob (`EXEC_YMSK=0` or `EXEC_COMPACT=1`) selects which
   one is in play. Compile-time sanity checks fire if it's set wrong.

2. **Elliptical geometry.** `i6`'s default geometry is a centered ellipse
   defined by `ACTIVE_RX_FRAC` × `ACTIVE_RY_FRAC` (a circle is the equal-axes
   case on a square grid). Added as `GEOMETRY_KIND=5 = GEOMETRY_ELLIPSE`.

A new section in `sim_config.h` (Section 9) documents the **two `UserData`
byte layouts** side by side — 88 bytes for ymsk mode, 96 bytes for compact
mode — so you can keep the `<main>.cu` / `jtv.cu` / `precond.cu` mirror
structs in sync.

## How to update the template when you write a new variant

The general procedure (which is what got us from v1 → v2):

1. **Read the new variant top-to-bottom.** Note every `#ifndef X #define X` block at
   the top and every `__constant__` value, every "geometry knob" used by the
   `BuildXxx()` host function, and every kernel-launch dimension.
2. **Diff against the existing template.**
   - **New knob in the variant?** → add an `#ifndef` block to the matching
     section of `sim_config.h` and a `?=` line plus `-D` flag to `params.mk`.
   - **New "kind" (geometry / IC / physics law)?** → extend the matching
     enum (`GEOMETRY_KIND`, `IC_KIND`, `ANISO_KIND`, …) with a new code
     and document it in the comment block above the enum.
   - **New struct field in `UserData`?** → record the offset in Section 9
     of `sim_config.h` and remind the user that `jtv.cu` and `precond.cu`
     must mirror it byte-for-byte.
   - **New launch shape (e.g. 1D vs 2D, different block size)?** → add the
     dim macros to Section 5 of `sim_config.h`. Don't remove old ones —
     other variants still need them.
3. **Add a sanity-check `#error`** in Section 8 for any new constraint
   (e.g. inner < outer for ring, both fractions > 0 for ellipse).
4. **Update `make show-config`** in `params.mk` to print the new knobs.
5. **Add a cookbook recipe** to this README so future readers can reproduce
   the variant with a one-liner.

The same pattern applies to *any* future variant. The point of the template
is that adding `i7`, `i8`, … should be additive — no existing knob's default
changes, no old recipe breaks.

## Quick start

1. Drop `sim_config.h` into the variant folder you want to clone, e.g.
   `src/irregular/2d_fft/2d_i6/`.
2. At the very top of the main `.cu` file add `#include "sim_config.h"` and
   delete that file's per-variant `#ifndef KRYLOV_DIM …` knob block.
3. Drop `params.mk` next to the variant's `Makefile` and add `include
   params.mk` near the top, deleting the old per-knob `?=` block.
4. Rebuild: `make show-config && make clean && make`.

## What each section in `sim_config.h` controls

| Section | Knobs |
|---|---|
| **0. Discretization** | `NX_VAL`, `NY_VAL`. `ng = NX_VAL/3`, `ncell = ng*ny`, `neq = 3*ncell`. |
| **0.5. Execution model** | `EXECUTION_MODEL` = `EXEC_YMSK` (i1..i5) or `EXEC_COMPACT` (i6). Determines `UserData` layout, kernel signatures, and which block-dim knobs apply. |
| **1. Geometry** | `GEOMETRY_KIND` picks bulk / square hole / ring / polycrystal / custom / **ellipse**. Each picks up a different sub-block of knobs. |
| **2. Initial condition** | `IC_KIND` picks uniform / head-on stripes / two-domain / grain-bumps / custom. |
| **3. Physics** | exchange `c_che`, anisotropy `c_chk`/`c_cha` and axis `c_msk`, DMI `c_chb` and dir `c_nsk`, damping `c_alpha`, gyromagnetic `c_chg`, applied field `happ_*`. |
| **4. Demag** | `DEMAG_STRENGTH` (0 disables), `DEMAG_THICK`, optional `DEMAG_WINDOWED` (only for polycrystal `2d_i3`). |
| **5. Solver** | `RTOL_VAL`, `ATOL_VAL`, `T_TOTAL`, `MAX_BDF_ORDER`, `KRYLOV_DIM`, `BLOCK_X`/`BLOCK_Y` (ymsk mode), **`BLOCK_SIZE`** (compact mode), `GS_KIND`. |
| **6. Output** | `ENABLE_OUTPUT`, `EARLY_SAVE_UNTIL/EVERY`, `LATE_SAVE_EVERY`, `WRITE_FINAL_STATE`. |
| **9. UserData layouts** | Reference: byte-by-byte struct layout for both execution models. |

## Cookbook: reproducing each variant

### Reproduce `2d_i1` — head-on three-stripe domain wall (ymsk)

```
make EXECUTION_MODEL=0 \
     NX_VAL=1536 NY_VAL=512 \
     GEOMETRY_KIND=0  IC_KIND=1 \
     STRIPE_LEFT_FRAC=0.25 STRIPE_RIGHT_FRAC=0.75 \
     INIT_RANDOM_EPS=0.01 INIT_RANDOM_SEED=12345 \
     PHYS_C_CHK=4.0 PHYS_C_CHE=4.0 PHYS_C_CHA=0.0 \
     ANISO_KIND=0  PHYS_MSK_X=1 PHYS_MSK_Y=0 PHYS_MSK_Z=0 \
     PHYS_C_CHB=0.3  PHYS_NSK_X=1 PHYS_NSK_Y=0 PHYS_NSK_Z=0 \
     DEMAG_STRENGTH=2.0 DEMAG_THICK=1.0
```

### Reproduce `2d_i2` — square hole, uniform IC (ymsk)

```
make EXECUTION_MODEL=0 \
     GEOMETRY_KIND=1  IC_KIND=0 \
     HOLE_CENTER_X_FRAC=0.5 HOLE_CENTER_Y_FRAC=0.5 HOLE_RADIUS_FRAC_Y=0.05 \
     INIT_MX=1.0 INIT_MY=-0.0175 INIT_MZ=0.0 \
     PHYS_C_CHK=1.0 PHYS_C_CHE=50.0 \
     ANISO_KIND=1  PHYS_MSK_X=1 PHYS_MSK_Y=1 PHYS_MSK_Z=1 \
     DEMAG_STRENGTH=4.0 DEMAG_THICK=1.0
```

### Reproduce `2d_i3` — polycrystal (ymsk + per-cell easy-axis tables)

```
make EXECUTION_MODEL=0 \
     GEOMETRY_KIND=3  IC_KIND=3 \
     NUM_GRAINS=72 DEAD_GRAIN_FRAC=0.16 \
     HOLE_SEED=20251104 MASK_EPS_CELLS=2.2 \
     GRAIN_Z_BIAS=1.6 IC_CORE_MZ=0.95 \
     PHYS_C_CHE=4.0 PHYS_C_CHK=1.0 \
     ANISO_KIND=0 PHYS_C_CHB=0.6 \
     DEMAG_STRENGTH=1.0 DEMAG_WINDOWED=1
```

### Reproduce `2d_i4` — square hole + applied field + two-domain IC (ymsk)

```
make EXECUTION_MODEL=0 \
     GEOMETRY_KIND=1  IC_KIND=2 \
     HOLE_CENTER_X_FRAC=0.5 HOLE_CENTER_Y_FRAC=0.5 HOLE_RADIUS_FRAC_Y=0.05 \
     TWO_DOMAIN_SPLIT_FRAC=0.875 TWO_DOMAIN_TAIL_MY=0.0175 \
     PHYS_C_CHK=1.0 PHYS_C_CHE=20.0 \
     ANISO_KIND=1 PHYS_MSK_X=1 PHYS_MSK_Y=1 PHYS_MSK_Z=1 \
     HAPP_ENABLE=1 HAPP_X=-0.2 HAPP_Y=0 HAPP_Z=0 \
     DEMAG_STRENGTH=4.0 DEMAG_THICK=1.0
```

### Reproduce `2d_i5` — ring (ymsk)

```
make EXECUTION_MODEL=0 \
     GEOMETRY_KIND=2  IC_KIND=0 \
     OUTER_W_FRAC=0.5 OUTER_H_FRAC=0.5 \
     INNER_W_FRAC_OF_OUTER=0.5 INNER_H_FRAC_OF_OUTER=0.5 \
     PHYS_C_CHK=1.0 PHYS_C_CHE=6.0 \
     ANISO_KIND=1 PHYS_MSK_X=1 PHYS_MSK_Y=1 PHYS_MSK_Z=1 \
     DEMAG_STRENGTH=4.0 DEMAG_THICK=1.0
```

### Reproduce `2d_i6` — ellipse / circle (compact)  ⭐ new

```
# Default i6: ellipse with rx=0.25*ng, ry=0.50*ny
make EXECUTION_MODEL=1 \
     GEOMETRY_KIND=5  IC_KIND=0 \
     ACTIVE_RX_FRAC=0.25 ACTIVE_RY_FRAC=0.50 \
     PHYS_C_CHE=10.0 PHYS_C_CHK=1.0 \
     ANISO_KIND=1 PHYS_MSK_X=1 PHYS_MSK_Y=1 PHYS_MSK_Z=1 \
     DEMAG_STRENGTH=4.0 DEMAG_THICK=1.0 \
     BLOCK_SIZE=256

# Circle on a square grid (rx == ry):
make EXECUTION_MODEL=1 GEOMETRY_KIND=5 \
     ACTIVE_RX_FRAC=0.25 ACTIVE_RY_FRAC=0.25 \
     NX_VAL=1536 NY_VAL=512 BLOCK_SIZE=256
```

> Note: the `ACTIVE_R{X,Y}_FRAC` axes are independent — they're fractions of
> `ng` and `ny` respectively. On a square grid (`ng = ny`) and equal
> fractions you get a circle; on the rectangular default grid (`ng=512`,
> `ny=512` after `NX_VAL/3`) equal fractions also give a circle, but on
> `ng != ny` they give an ellipse.

## Adding a new geometry (the recommended way)

The architecture's whole geometry encoding lives in **one** structure
(either `d_ymsk` or `d_active_ids`/`d_inactive_ids`), so adding a new
shape is purely a host-side construction.

### EXEC_YMSK (i1..i5 style)

1. Set `GEOMETRY_KIND = GEOMETRY_CUSTOM` (= 4).
2. In `main()` of the variant's main `.cu`, allocate `h_ymsk[3*ncell]`,
   fill with your shape (`1.0` = active, `0.0` = hole), upload to `d_ymsk`.
3. Hole cells produce `0` outputs because every kernel multiplies by `ymsk`.

### EXEC_COMPACT (i6 style — recommended)

1. Set `GEOMETRY_KIND = GEOMETRY_CUSTOM` (= 4) and
   `EXECUTION_MODEL = EXEC_COMPACT` (= 1).
2. In `main()`, scan all cells, fill two arrays:
   `h_active_ids[n_active]` (cell ids inside your shape) and
   `h_inactive_ids[n_inactive]` (the rest). Upload both.
3. Initial condition: `m=0` at all hole cells; CVODE preserves it.
4. No kernel changes needed — the existing compact kernels read
   `active_ids[tid]` and skip everything else.

For sparse geometries (active fraction ≪ 1) **the compact path is
substantially faster** — threads in holes simply don't exist, instead of
running and producing `output * 0`.

## Things that look like knobs but are NOT

These you'll have to edit in source if you really need them:

| What | Where | Why it's not a macro |
|---|---|---|
| Periodic vs Dirichlet BCs | `wrap_x` / `wrap_y` in `<main>.cu`, jtv.cu, precond.cu | Affects every kernel; not a one-liner toggle. |
| The byte layout of `UserData` | `<main>.cu`, jtv.cu (`JtvUserData`), precond.cu (`PcUserData`) | Three structs must stay byte-for-byte mirrors. Extending requires a synchronized change in all three. See Section 9 of `sim_config.h` for the two layouts. |
| Material constants in jtv.cu and precond.cu (`jc_*`, `pc_*`) | Top of those files | They are `__constant__` device variables. Changing physics in `sim_config.h` requires updating these mirrors. |
| `FusedNVec` linear-combination override | deferred_nvector.cu | i6 hardwires this off (profile-proven L2 thrashing); not a runtime toggle. |

A clean follow-up refactor is to have jtv.cu and precond.cu also `#include
"sim_config.h"` and use `PHYS_C_CHK` etc. directly in the `__constant__`
declarations. That eliminates the mirror-maintenance problem entirely.

## Sanity checks built in

`sim_config.h` `#error`s out at compile time if:

- `NX_VAL` is not a multiple of `GROUPSIZE` (= 3),
- `BLOCK_X * BLOCK_Y > 1024` or `BLOCK_SIZE > 1024` (CUDA per-block thread limit),
- `EXECUTION_MODEL` is neither `EXEC_YMSK` nor `EXEC_COMPACT`,
- `GEOMETRY_KIND == GEOMETRY_POLYCRYSTAL` and `NUM_GRAINS < 1`,
- `GEOMETRY_KIND == GEOMETRY_RING` and inner ≥ outer fractions,
- `GEOMETRY_KIND == GEOMETRY_ELLIPSE` and either fraction ≤ 0.

`make show-config` prints the full configuration — including which
execution model is active and which block-dim knobs apply — before building.
