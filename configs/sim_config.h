/* ============================================================================
 * sim_config.h  —  General input template for gpu-micromagnetics-irregular
 *
 * Centralized configuration header that consolidates EVERY knob across the
 * 2d_i1 ... 2d_i5 variants of the project. Drop this file at the top of
 * 2d_fft.cu (and #include it before any other module header), and override
 * any value you like via:
 *
 *     1. Editing the defaults in this file directly, OR
 *     2. Compile-time -D flags from the Makefile (preferred for sweeps), OR
 *     3. A separate tiny "case header" you #include before this one that
 *        #defines just the knobs you want to change.
 *
 * Every macro below is wrapped in `#ifndef ... #define ... #endif` so that
 * Makefile -D overrides always win.
 *
 * ─── HOW TO USE FOR A NEW PROBLEM ─────────────────────────────────────
 * Step 1.  Pick a geometry preset:
 *            GEOMETRY_BULK         -- no holes, fully periodic body
 *            GEOMETRY_HOLE_SQUARE  -- one square interior hole
 *            GEOMETRY_RING         -- outer rect minus inner rect (ring)
 *            GEOMETRY_POLYCRYSTAL  -- Voronoi grains + dead-grain holes
 *            GEOMETRY_CUSTOM       -- you build h_ymsk[] yourself
 *
 *          Set GEOMETRY_KIND below (or pass -DGEOMETRY_KIND=... in the
 *          Makefile). Then fill in the matching geometry-specific knobs
 *          in the corresponding section.
 *
 * Step 2.  Pick an initial-condition preset:
 *            IC_UNIFORM            -- all active cells in one direction
 *            IC_HEAD_ON_STRIPES    -- 3-stripe domain wall along x (i1)
 *            IC_TWO_DOMAIN         -- upper anti-aligned, lower aligned (i4)
 *            IC_GRAIN_BUMPS        -- per-grain core ±z, blend at edges (i3)
 *            IC_CUSTOM             -- you fill ydata[] yourself
 *
 * Step 3.  Set physics constants (anisotropy, exchange, DMI, demag,
 *          applied field). Defaults here match the most general
 *          square-hole + uniform-IC variant (2d_i2 / 2d_i4 baseline).
 *
 * Step 4.  Tune solver & kernel settings if needed (RTOL, KRYLOV_DIM,
 *          BLOCK_X/Y, T_TOTAL).
 *
 * Step 5.  `make clean && make` and run.
 *
 * ─── INVARIANTS YOU MUST NOT BREAK ────────────────────────────────────
 *   - NX_VAL must be a multiple of GROUPSIZE (= 3). The "logical" number of
 *     columns the kernels see is ng = NX_VAL / GROUPSIZE.
 *   - The UserData struct in 2d_fft.cu, JtvUserData in jtv.cu and
 *     PcUserData in precond.cu MUST stay byte-compatible (mirror layout).
 *     If you add a field, add it to all three in the same offset.
 *   - The material constants below are also baked into __constant__ memory
 *     in jtv.cu and precond.cu. If you change c_chk / c_che / c_alpha /
 *     c_chg / c_chb, you MUST update the matching jc_* and pc_* values.
 *   - Hole / inactive cells live entirely in the SoA mask d_ymsk; every
 *     kernel multiplies its output by ymsk[mα]. Do NOT add `if(active)`
 *     branches in the kernels — build h_ymsk on the host and rely on it.
 * ============================================================================ */

#ifndef SIM_CONFIG_H
#define SIM_CONFIG_H

/* ═════════════════════════════════════════════════════════════════════
 * SECTION 0 — Discretization / problem size
 * ═════════════════════════════════════════════════════════════════════ */

/* GROUPSIZE: physical cells per "x-stride" (do not change unless you also
 * rewrite the SoA indexing in every kernel). Always 3 for this project. */
#define GROUPSIZE 3

/* Grid dimensions in cells. Scaling logs in 2d_i1 swept from 96x32 up to
 * 3072x1024; FFT crossover vs O(N^2) direct happens around 384x128. */
#ifndef NX_VAL
#define NX_VAL 1536            /* must be multiple of GROUPSIZE */
#endif
#ifndef NY_VAL
#define NY_VAL 512
#endif

/* Periodic boundary conditions are always on in both x and y — change
 * this only by editing wrap_x / wrap_y in 2d_fft.cu and friends. */


/* ═════════════════════════════════════════════════════════════════════
 * SECTION 1 — Geometry
 * Pick exactly ONE GEOMETRY_KIND value. Hole/active cells are encoded
 * in the SoA mask d_ymsk (1 = active, 0 = hole) which the host fills
 * before launch and which every kernel reads transparently.
 * ═════════════════════════════════════════════════════════════════════ */

#define GEOMETRY_BULK         0   /* no holes — fully active body         */
#define GEOMETRY_HOLE_SQUARE  1   /* one centered square interior hole    */
#define GEOMETRY_RING         2   /* outer rect minus inner rect (ring)   */
#define GEOMETRY_POLYCRYSTAL  3   /* Voronoi grains + dead-grain holes    */
#define GEOMETRY_CUSTOM       4   /* you build h_ymsk[] yourself          */

#ifndef GEOMETRY_KIND
#define GEOMETRY_KIND GEOMETRY_HOLE_SQUARE
#endif

/* ── Square-hole knobs (GEOMETRY_HOLE_SQUARE) ────────────────────────
 * The hole is a square centered at (HOLE_CENTER_X_FRAC, HOLE_CENTER_Y_FRAC)
 * (fractions of the ng × ny grid). HOLE_RADIUS_FRAC_Y is the HALF-SIDE
 * of the square in units of ny (so the full side is 2·HOLE_RADIUS_FRAC_Y·ny).
 * Macro name kept for backward compatibility with sweep targets. */
#ifndef HOLE_CENTER_X_FRAC
#define HOLE_CENTER_X_FRAC 0.50
#endif
#ifndef HOLE_CENTER_Y_FRAC
#define HOLE_CENTER_Y_FRAC 0.50
#endif
#ifndef HOLE_RADIUS_FRAC_Y
#define HOLE_RADIUS_FRAC_Y 0.22
#endif

/* ── Ring knobs (GEOMETRY_RING, ie 2d_i5) ────────────────────────────
 * Outer active rectangle, centered:
 *     width  = OUTER_W_FRAC * ng
 *     height = OUTER_H_FRAC * ny
 * Inner rect hole, centered inside outer:
 *     width  = INNER_W_FRAC_OF_OUTER * outer_w
 *     height = INNER_H_FRAC_OF_OUTER * outer_h */
#ifndef OUTER_W_FRAC
#define OUTER_W_FRAC 0.5
#endif
#ifndef OUTER_H_FRAC
#define OUTER_H_FRAC 0.5
#endif
#ifndef INNER_W_FRAC_OF_OUTER
#define INNER_W_FRAC_OF_OUTER 0.5
#endif
#ifndef INNER_H_FRAC_OF_OUTER
#define INNER_H_FRAC_OF_OUTER 0.5
#endif

/* ── Polycrystal knobs (GEOMETRY_POLYCRYSTAL, ie 2d_i3) ──────────────
 * NUM_GRAINS Voronoi seeds (stratified jitter, periodic), DEAD_GRAIN_FRAC
 * of them are killed → polygonal holes with soft tanh boundary of width
 * MASK_EPS_CELLS. */
#ifndef NUM_GRAINS
#define NUM_GRAINS 72
#endif
#ifndef DEAD_GRAIN_FRAC
#define DEAD_GRAIN_FRAC 0.16
#endif
#ifndef HOLE_SEED
#define HOLE_SEED 20251104
#endif
#ifndef MASK_EPS_CELLS
#define MASK_EPS_CELLS 2.2          /* tanh boundary width, in cell units */
#endif


/* ═════════════════════════════════════════════════════════════════════
 * SECTION 2 — Initial condition
 * Pick exactly ONE IC_KIND value.
 * ═════════════════════════════════════════════════════════════════════ */

#define IC_UNIFORM           0   /* all active cells m = (INIT_MX, INIT_MY, INIT_MZ) */
#define IC_HEAD_ON_STRIPES   1   /* three-stripe head-on transition along x (i1) */
#define IC_TWO_DOMAIN        2   /* upper anti-aligned + lower aligned (i4) */
#define IC_GRAIN_BUMPS       3   /* per-grain core ±z bumps, polycrystal only (i3) */
#define IC_CUSTOM            4   /* fill ydata[] yourself in main() */

#ifndef IC_KIND
#define IC_KIND IC_UNIFORM
#endif

/* ── Uniform IC (IC_UNIFORM) ─────────────────────────────────────────
 * After normalization the magnitude is 1 in active cells and 0 in holes.
 * A small tilt off the easy axis (default INIT_MY = -0.0175) breaks
 * ±x symmetry so demag + geometry can drive non-uniform dynamics. */
#ifndef INIT_MX
#define INIT_MX 1.0
#endif
#ifndef INIT_MY
#define INIT_MY -0.0175
#endif
#ifndef INIT_MZ
#define INIT_MZ 0.0
#endif

/* ── Head-on three-stripe IC (IC_HEAD_ON_STRIPES) ────────────────────
 *   i ∈ [0,                   ng·STRIPE_LEFT_FRAC):  mx = -1
 *   i ∈ [ng·STRIPE_LEFT_FRAC, ng·STRIPE_RIGHT_FRAC): mx = +1
 *   i ∈ [ng·STRIPE_RIGHT_FRAC, ng):                  mx = -1
 * INIT_RANDOM_EPS sets a uniform random perturbation in (my, mz) to
 * break y-z symmetry; INIT_RANDOM_SEED makes it reproducible. */
#ifndef STRIPE_LEFT_FRAC
#define STRIPE_LEFT_FRAC  0.25
#endif
#ifndef STRIPE_RIGHT_FRAC
#define STRIPE_RIGHT_FRAC 0.75
#endif
#ifndef INIT_RANDOM_EPS
#define INIT_RANDOM_EPS   0.01
#endif
#ifndef INIT_RANDOM_SEED
#define INIT_RANDOM_SEED  12345
#endif

/* ── Two-domain IC (IC_TWO_DOMAIN, ie 2d_i4) ─────────────────────────
 *   gy >= TWO_DOMAIN_SPLIT_FRAC * ny : m = (-1, 0, 0)
 *   gy <  TWO_DOMAIN_SPLIT_FRAC * ny : m = (+1, ±TWO_DOMAIN_TAIL_MY, 0)
 * with sign of m_y chosen by left/right half. */
#ifndef TWO_DOMAIN_SPLIT_FRAC
#define TWO_DOMAIN_SPLIT_FRAC 0.875   /* (7*ny)/8 in i4 default */
#endif
#ifndef TWO_DOMAIN_TAIL_MY
#define TWO_DOMAIN_TAIL_MY    0.0175  /* uses INIT_MY-style amplitude */
#endif

/* ── Polycrystal grain bumps (IC_GRAIN_BUMPS, ie 2d_i3) ──────────────
 * Per-grain z-bias, with mz ramped down as a function of distance to
 * the grain seed; in-plane tangent rotates around the seed and blends
 * toward the grain easy axis as you approach the edge. */
#ifndef GRAIN_Z_BIAS
#define GRAIN_Z_BIAS 1.6
#endif
#ifndef IC_CORE_MZ
#define IC_CORE_MZ   0.95
#endif


/* ═════════════════════════════════════════════════════════════════════
 * SECTION 3 — Physics constants
 *
 * IMPORTANT: these macros set the values at the top of 2d_fft.cu, but
 * jtv.cu and precond.cu store their own __constant__ copies (jc_*, pc_*).
 * If you change anything here you MUST also update those mirror files,
 * or move all three to read from the same header. The cleanest fix is
 * to refactor those files to #include this header and use these macros
 * directly — left as a TODO so this file remains a drop-in addition.
 * ═════════════════════════════════════════════════════════════════════ */

/* Damping & gyromagnetic ratio. Standard simplified LLG:
 *     dm/dt = c_chg (m × h) + c_alpha ( h − (m·h) m ) */
#ifndef PHYS_C_CHG
#define PHYS_C_CHG    1.0     /* gyromagnetic precession coefficient */
#endif
#ifndef PHYS_C_ALPHA
#define PHYS_C_ALPHA  0.2     /* Gilbert damping                     */
#endif

/* Exchange constant (coefficient on the 5-point neighbor sum).
 * Range across variants: 4 (i1, i3) ... 50 (i2). Larger = stiffer wall. */
#ifndef PHYS_C_CHE
#define PHYS_C_CHE    50.0
#endif

/* Anisotropy strength. Range across variants: 1 (i2/i4/i5) ... 4 (i1/i3).
 * Two anisotropy laws are used in the project:
 *   - Linear (i1, i3):  h_α += msk_α · (c_chk·m1 + c_cha)
 *   - Cubic  (i2/4/5):  h_α += msk_α · c_chk · m_α · (m_α² − 1)
 * Pick via ANISO_KIND below. */
#ifndef PHYS_C_CHK
#define PHYS_C_CHK    1.0
#endif
#ifndef PHYS_C_CHA
#define PHYS_C_CHA    0.0
#endif

#define ANISO_LINEAR  0   /* h += msk · (chk·m1 + cha)         (i1, i3)  */
#define ANISO_CUBIC   1   /* h += msk_α · chk · m_α·(m_α²-1)   (i2/4/5)  */

#ifndef ANISO_KIND
#define ANISO_KIND ANISO_CUBIC
#endif

/* Easy-axis direction c_msk. Many variants use {1,0,0} (easy x); 2d_i2/i4/i5
 * use {1,1,1} together with the cubic law. For per-cell axis (polycrystal),
 * leave this as a fallback and rely on udata->d_msk built on the host. */
#ifndef PHYS_MSK_X
#define PHYS_MSK_X 1.0
#endif
#ifndef PHYS_MSK_Y
#define PHYS_MSK_Y 0.0
#endif
#ifndef PHYS_MSK_Z
#define PHYS_MSK_Z 0.0
#endif

/* DMI strength and direction.
 *   c_chb is the magnitude (default 0.3 across variants).
 *   c_nsk = (PHYS_NSK_X, PHYS_NSK_Y, PHYS_NSK_Z) is the direction. */
#ifndef PHYS_C_CHB
#define PHYS_C_CHB 0.3
#endif
#ifndef PHYS_NSK_X
#define PHYS_NSK_X 1.0
#endif
#ifndef PHYS_NSK_Y
#define PHYS_NSK_Y 0.0
#endif
#ifndef PHYS_NSK_Z
#define PHYS_NSK_Z 0.0
#endif

/* Uniform applied (Zeeman) field. Set HAPP_ENABLE = 1 to add it to h_total
 * (only 2d_i4 uses this). The components are absolute, NOT scaled. */
#ifndef HAPP_ENABLE
#define HAPP_ENABLE 0
#endif
#ifndef HAPP_X
#define HAPP_X (-0.2)
#endif
#ifndef HAPP_Y
#define HAPP_Y 0.0
#endif
#ifndef HAPP_Z
#define HAPP_Z 0.0
#endif


/* ═════════════════════════════════════════════════════════════════════
 * SECTION 4 — Demagnetization (cuFFT D2Z/Z2D pipeline)
 *
 * Demag tensor is precomputed once in Demag_Init using the closed-form
 * Newell calt/ctt construction (81-pt face-averaged). Stored permanently
 * on device as half-spectrum (9 components × ny·(nx/2+1) complex).
 * Per RHS evaluation:
 *   gather → cuFFT D2Z (batch=3) → multiply f̂·m̂ → cuFFT Z2D (batch=3)
 *   → scatter (FFT-shift + scale).
 *
 * DEMAG_STRENGTH = 0.0  → disables demag (h_dmag buffer stays zero)
 * DEMAG_STRENGTH > 0.0  → enables FFT demag, scales N(0) accordingly
 * DEMAG_THICK           → cell thickness in z (in cell-spacing units)
 * ═════════════════════════════════════════════════════════════════════ */

#ifndef DEMAG_STRENGTH
#define DEMAG_STRENGTH 4.0
#endif
#ifndef DEMAG_THICK
#define DEMAG_THICK    1.0
#endif

/* Windowed demag (only used in 2d_i3 polycrystal): if nonzero, the gather
 * kernel multiplies y by the per-cell weight w on the fly so that demag
 * sees the windowed magnetization w·m. Requires udata->d_w to be allocated
 * and Demag_ApplyWindowed to be linked. Leave 0 for hole/ring/uniform. */
#ifndef DEMAG_WINDOWED
#define DEMAG_WINDOWED 0
#endif


/* ═════════════════════════════════════════════════════════════════════
 * SECTION 5 — CVODE / SPGMR / kernel tuning
 * ═════════════════════════════════════════════════════════════════════ */

/* Time integration */
#ifndef T_TOTAL
#define T_TOTAL 1000.0          /* end time of the simulation */
#endif

/* CVODE BDF order cap. 5 is the SUNDIALS maximum. */
#ifndef MAX_BDF_ORDER
#define MAX_BDF_ORDER 5
#endif

/* Tolerances. RTOL = relative; ATOL = absolute (broadcast to all components).
 * Range across variants: 1e-4 ... 1e-6. Tighter → more steps, more accurate. */
#ifndef RTOL_VAL
#define RTOL_VAL 1.0e-4
#endif
#ifndef ATOL_VAL
#define ATOL_VAL 1.0e-4
#endif

/* SPGMR Krylov subspace dimension.
 *   0   → SUNDIALS default (= min(neq, 5))
 *   5   → most variants (best at small/medium grids per i1 sweeps)
 *  10..30 → more headroom before restart, useful at very small grids
 *           where neq < 10k and convergence stalls. */
#ifndef KRYLOV_DIM
#define KRYLOV_DIM 5
#endif

/* CUDA block dimensions for the unified RHS / Jv / preconditioner kernels.
 * BLOCK_X * BLOCK_Y must be ≤ 1024. Default 16x8 = 128 threads/block.
 * The polycrystal tiled stencil sizes its smem off these (BLOCK_X+2)·(BLOCK_Y+2). */
#ifndef BLOCK_X
#define BLOCK_X 16
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 8
#endif

/* Gram-Schmidt orthogonalization in SPGMR.
 *   Small problems (neq < ~500k): Classical GS (CGS) — overhead-limited,
 *     1 fused dot-product-multi call per orthogonalization. Fastest.
 *   Large problems: Modified GS (MGS) — bandwidth-limited, more stable
 *     numerically but has a sync per dot. The default 2d_fft.cu picks
 *     based on neq; you can force one mode here.
 *
 *   GS_KIND_AUTO    — pick CGS if neq < 500k, MGS otherwise (default)
 *   GS_KIND_CGS     — always Classical
 *   GS_KIND_MGS     — always Modified */
#define GS_KIND_AUTO 0
#define GS_KIND_CGS  1
#define GS_KIND_MGS  2

#ifndef GS_KIND
#define GS_KIND GS_KIND_AUTO
#endif


/* ═════════════════════════════════════════════════════════════════════
 * SECTION 6 — Output / I/O schedule
 * ═════════════════════════════════════════════════════════════════════ */

/* ENABLE_OUTPUT = 1 turns on per-frame dumps to output.txt. Zero by
 * default — you almost never want output during sweeps. */
#ifndef ENABLE_OUTPUT
#define ENABLE_OUTPUT 0
#endif

/* Two-rate save schedule. Early frames (when domain wall / nucleation
 * dynamics are fastest) are saved more often; late steady-state frames
 * are saved sparsely. */
#ifndef EARLY_SAVE_UNTIL
#define EARLY_SAVE_UNTIL 80.0   /* time in simulation units */
#endif
#ifndef EARLY_SAVE_EVERY
#define EARLY_SAVE_EVERY 5      /* save every N output steps for t ≤ EARLY_SAVE_UNTIL */
#endif
#ifndef LATE_SAVE_EVERY
#define LATE_SAVE_EVERY  100    /* save every N output steps for t >  EARLY_SAVE_UNTIL */
#endif

/* Final-state dump (writes the last frame to output.txt regardless of
 * ENABLE_OUTPUT). Useful for post-mortem visualization. */
#ifndef WRITE_FINAL_STATE
#define WRITE_FINAL_STATE 1
#endif


/* ═════════════════════════════════════════════════════════════════════
 * SECTION 7 — Derived constants (do not edit)
 * ═════════════════════════════════════════════════════════════════════ */

/* SUNDIALS-typed copies of RTOL/ATOL/T_TOTAL. */
#define RTOL  SUN_RCONST(RTOL_VAL)
#define ATOL1 SUN_RCONST(ATOL_VAL)
#define ATOL2 SUN_RCONST(ATOL_VAL)
#define ATOL3 SUN_RCONST(ATOL_VAL)
#define T0    SUN_RCONST(0.0)
#define T1    SUN_RCONST(0.1)              /* output cadence */
#define ZERO  SUN_RCONST(0.0)


/* ═════════════════════════════════════════════════════════════════════
 * SECTION 8 — Sanity checks (compile-time)
 * ═════════════════════════════════════════════════════════════════════ */

#if (NX_VAL % GROUPSIZE) != 0
#  error "NX_VAL must be a multiple of GROUPSIZE (=3)."
#endif

#if (BLOCK_X * BLOCK_Y) > 1024
#  error "BLOCK_X * BLOCK_Y exceeds the CUDA per-block thread limit (1024)."
#endif

#if (GEOMETRY_KIND == GEOMETRY_POLYCRYSTAL) && (NUM_GRAINS < 1)
#  error "GEOMETRY_POLYCRYSTAL requires NUM_GRAINS >= 1."
#endif

#if (GEOMETRY_KIND == GEOMETRY_RING) && \
    ((INNER_W_FRAC_OF_OUTER >= 1.0) || (INNER_H_FRAC_OF_OUTER >= 1.0))
#  error "Ring geometry requires inner < outer (INNER_*_FRAC_OF_OUTER < 1.0)."
#endif

#endif /* SIM_CONFIG_H */
