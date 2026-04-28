/* ============================================================================
 * sim_config.h  —  General input template for gpu-micromagnetics-irregular
 *
 * v2 (covers 2d_i1 ... 2d_i6).
 *
 * Centralized configuration header that consolidates EVERY knob across all
 * variants.  Drop this file at the top of the variant's main .cu (and
 * #include it before any other module header), and override any value you
 * like via:
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
 *
 * Step 0.  Pick an EXECUTION MODEL.  This is the BIGGEST architectural
 *          fork in the project (introduced with 2d_i6):
 *
 *            EXEC_YMSK    -- byte mask d_ymsk[3*ncell] (1 = active, 0 = hole),
 *                            kernels launch over the FULL grid and multiply
 *                            their output by ymsk.  Used by i1..i5.
 *                            UserData layout: 88 bytes (no id lists).
 *                            Block dims: BLOCK_X x BLOCK_Y  (2D launches).
 *
 *            EXEC_COMPACT -- compact index lists d_active_ids[n_active] and
 *                            d_inactive_ids[n_inactive], kernels launch
 *                            ONE THREAD PER ACTIVE CELL.  Used by i6.
 *                            UserData layout: 96 bytes (adds id lists +
 *                            n_active + n_inactive + 4 B pad).
 *                            Block dims: BLOCK_SIZE (1D launches).
 *
 *          Which one to pick?
 *          - >50% active cells (compact body, small holes):
 *            ymsk and compact are within ~10% of each other.  Either works.
 *          - <50% active cells (dot, ring, ellipse, sparse polycrystal):
 *            COMPACT wins — sometimes by 3-5x — because threads in holes
 *            simply don't exist.
 *          - For new variants you write from scratch, prefer COMPACT;
 *            the indexing is simpler (no `if (mask)` mental model needed).
 *
 *          The two models cannot be mixed in one variant — they imply
 *          different UserData byte layouts and different kernel signatures.
 *
 * Step 1.  Pick a GEOMETRY preset (Section 1).
 * Step 2.  Pick an INITIAL CONDITION preset (Section 2).
 * Step 3.  Set physics constants (Section 3).
 * Step 4.  Tune solver & kernel settings (Section 5).
 * Step 5.  `make clean && make show-config && make` and run.
 *
 * ─── INVARIANTS YOU MUST NOT BREAK ────────────────────────────────────
 *   - NX_VAL must be a multiple of GROUPSIZE (= 3).  ng = NX_VAL / GROUPSIZE.
 *   - The UserData / JtvUserData / PcUserData structs in <main>.cu, jtv.cu,
 *     and precond.cu MUST stay byte-compatible mirrors of each other.
 *     The layout depends on EXECUTION_MODEL — see Section 9 below for
 *     the exact field order in each mode.
 *   - The material constants below (PHYS_*) are also baked into __constant__
 *     memory in jtv.cu (jc_*) and precond.cu (pc_*).  Three mirrors that
 *     MUST stay in sync.  The cleanest fix is to refactor those files to
 *     #include this header and use these macros directly — left as TODO so
 *     this file remains a drop-in addition.
 *   - In EXEC_YMSK mode: hole/inactive cells live entirely in d_ymsk;
 *     every kernel multiplies its output by ymsk[mα].  Do NOT add
 *     `if (active)` branches.
 *   - In EXEC_COMPACT mode: hole cells are skipped entirely (kernels never
 *     run there).  ydot / Jv / z at hole entries are zeroed by small
 *     dedicated kernels at the start of f / JtvProduct / PrecondSolve so
 *     SUNDIALS' inner products see consistent zeros.  y[hole] is set to 0
 *     once at init and stays 0 forever (CVODE's linear-combination updates
 *     preserve it).  Do NOT inject a stray nonzero IC into a hole cell.
 * ============================================================================ */

#ifndef SIM_CONFIG_H
#define SIM_CONFIG_H

/* ═════════════════════════════════════════════════════════════════════
 * SECTION 0 — Discretization / problem size
 * ═════════════════════════════════════════════════════════════════════ */

#define GROUPSIZE 3

#ifndef NX_VAL
#define NX_VAL 1536            /* must be multiple of GROUPSIZE */
#endif
#ifndef NY_VAL
#define NY_VAL 512
#endif


/* ═════════════════════════════════════════════════════════════════════
 * SECTION 0.5 — Execution model (NEW in v2 — covers i6)
 *
 * EXEC_YMSK    : byte-mask approach used by i1..i5.
 * EXEC_COMPACT : compact active-cell approach used by i6.
 *
 * This selector is documentation + sanity-check oriented.  The actual
 * choice of model is structural (it dictates UserData layout, kernel
 * signatures, and which BLOCK_* knobs apply).  Setting this macro just
 * makes the active branch in your sources unambiguous and lets the
 * sanity checks at the bottom verify that the right knobs are present.
 * ═════════════════════════════════════════════════════════════════════ */

#define EXEC_YMSK     0   /* i1..i5 */
#define EXEC_COMPACT  1   /* i6     */

#ifndef EXECUTION_MODEL
#define EXECUTION_MODEL EXEC_YMSK
#endif


/* ═════════════════════════════════════════════════════════════════════
 * SECTION 1 — Geometry
 * Pick exactly ONE GEOMETRY_KIND value.
 *
 * Active/hole cells are encoded EITHER in d_ymsk (ymsk mode) OR in
 * d_active_ids / d_inactive_ids (compact mode).  In both cases the
 * host fills the data structure before launch using exactly the same
 * geometry predicate, so the GEOMETRY_KIND macros below are
 * model-agnostic.
 * ═════════════════════════════════════════════════════════════════════ */

#define GEOMETRY_BULK         0   /* no holes — fully active body         */
#define GEOMETRY_HOLE_SQUARE  1   /* one centered square interior hole    */
#define GEOMETRY_RING         2   /* outer rect minus inner rect (ring)   */
#define GEOMETRY_POLYCRYSTAL  3   /* Voronoi grains + dead-grain holes    */
#define GEOMETRY_CUSTOM       4   /* you build h_ymsk[] or id-lists yourself */
#define GEOMETRY_ELLIPSE      5   /* centered ellipse (i6); circle when rx=ry on square grid */

#ifndef GEOMETRY_KIND
#define GEOMETRY_KIND GEOMETRY_HOLE_SQUARE
#endif

/* ── Square-hole knobs (GEOMETRY_HOLE_SQUARE) ────────────────────────
 * Hole is centered at (HOLE_CENTER_X_FRAC, HOLE_CENTER_Y_FRAC) of the
 * ng × ny grid; HOLE_RADIUS_FRAC_Y is the HALF-SIDE in units of ny. */
#ifndef HOLE_CENTER_X_FRAC
#define HOLE_CENTER_X_FRAC 0.50
#endif
#ifndef HOLE_CENTER_Y_FRAC
#define HOLE_CENTER_Y_FRAC 0.50
#endif
#ifndef HOLE_RADIUS_FRAC_Y
#define HOLE_RADIUS_FRAC_Y 0.22
#endif

/* ── Ring knobs (GEOMETRY_RING, ie 2d_i5) ──────────────────────────── */
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

/* ── Polycrystal knobs (GEOMETRY_POLYCRYSTAL, ie 2d_i3) ─────────────── */
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

/* ── Ellipse knobs (GEOMETRY_ELLIPSE, ie 2d_i6) ──────────────────────
 * Centered ellipse with semi-axes
 *     rx = ACTIVE_RX_FRAC * ng     (cells, along x)
 *     ry = ACTIVE_RY_FRAC * ny     (cells, along y)
 * Cells satisfying (dx/rx)^2 + (dy/ry)^2 <= 1 are active.
 *
 * Special case: ACTIVE_RX_FRAC == ACTIVE_RY_FRAC on a square ng × ny
 * grid produces a circle of radius ACTIVE_RX_FRAC * ng cells. */
#ifndef ACTIVE_RX_FRAC
#define ACTIVE_RX_FRAC 0.25
#endif
#ifndef ACTIVE_RY_FRAC
#define ACTIVE_RY_FRAC 0.25
#endif


/* ═════════════════════════════════════════════════════════════════════
 * SECTION 2 — Initial condition
 * ═════════════════════════════════════════════════════════════════════ */

#define IC_UNIFORM           0   /* m = (INIT_MX, INIT_MY, INIT_MZ) on active cells */
#define IC_HEAD_ON_STRIPES   1   /* three-stripe head-on transition along x (i1) */
#define IC_TWO_DOMAIN        2   /* upper anti-aligned + lower aligned (i4) */
#define IC_GRAIN_BUMPS       3   /* per-grain core ±z bumps, polycrystal only (i3) */
#define IC_CUSTOM            4   /* fill ydata[] yourself in main() */

#ifndef IC_KIND
#define IC_KIND IC_UNIFORM
#endif

/* ── Uniform IC ─────────────────────────────────────────────────────── */
#ifndef INIT_MX
#define INIT_MX 1.0
#endif
#ifndef INIT_MY
#define INIT_MY -0.0175
#endif
#ifndef INIT_MZ
#define INIT_MZ 0.0
#endif

/* ── Head-on three-stripe IC (i1) ───────────────────────────────────── */
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

/* ── Two-domain IC (i4) ─────────────────────────────────────────────── */
#ifndef TWO_DOMAIN_SPLIT_FRAC
#define TWO_DOMAIN_SPLIT_FRAC 0.875
#endif
#ifndef TWO_DOMAIN_TAIL_MY
#define TWO_DOMAIN_TAIL_MY    0.0175
#endif

/* ── Polycrystal grain bumps (i3) ───────────────────────────────────── */
#ifndef GRAIN_Z_BIAS
#define GRAIN_Z_BIAS 1.6
#endif
#ifndef IC_CORE_MZ
#define IC_CORE_MZ   0.95
#endif


/* ═════════════════════════════════════════════════════════════════════
 * SECTION 3 — Physics constants
 *
 * IMPORTANT: these macros set the values at the top of the main .cu,
 * but jtv.cu and precond.cu store their own __constant__ copies
 * (jc_*, pc_*).  If you change anything here you MUST also update
 * those mirror files.  See README "Things that look like knobs but
 * are NOT" for details and a recommended refactor.
 * ═════════════════════════════════════════════════════════════════════ */

#ifndef PHYS_C_CHG
#define PHYS_C_CHG    1.0
#endif
#ifndef PHYS_C_ALPHA
#define PHYS_C_ALPHA  0.2
#endif

/* Range across variants: 4 (i1, i3) ... 50 (i2).  i6 default = 10. */
#ifndef PHYS_C_CHE
#define PHYS_C_CHE    50.0
#endif

#ifndef PHYS_C_CHK
#define PHYS_C_CHK    1.0
#endif
#ifndef PHYS_C_CHA
#define PHYS_C_CHA    0.0
#endif

#define ANISO_LINEAR  0   /* h += msk · (chk·m1 + cha)         (i1, i3)  */
#define ANISO_CUBIC   1   /* h += msk_α · chk · m_α·(m_α²-1)   (i2/4/5/6) */

#ifndef ANISO_KIND
#define ANISO_KIND ANISO_CUBIC
#endif

#ifndef PHYS_MSK_X
#define PHYS_MSK_X 1.0
#endif
#ifndef PHYS_MSK_Y
#define PHYS_MSK_Y 0.0
#endif
#ifndef PHYS_MSK_Z
#define PHYS_MSK_Z 0.0
#endif

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

/* Uniform applied (Zeeman) field (i4 only).  HAPP_ENABLE = 1 to add. */
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
 * NOTE on EXEC_COMPACT mode: Demag_Apply is FULL-GRID by design (FFT
 * cannot be compacted).  For active cells the result is correct; for
 * hole cells h_dmag[] contains garbage, but the compact RHS only
 * reads h_dmag at active positions, so it doesn't matter.  Hole cells
 * contribute 0 to the input FFT (since y=0 there), so they don't
 * contaminate the active cells' output.
 * ═════════════════════════════════════════════════════════════════════ */

#ifndef DEMAG_STRENGTH
#define DEMAG_STRENGTH 4.0
#endif
#ifndef DEMAG_THICK
#define DEMAG_THICK    1.0
#endif
#ifndef DEMAG_WINDOWED
#define DEMAG_WINDOWED 0     /* polycrystal-only: gather pre-multiplies y by w */
#endif


/* ═════════════════════════════════════════════════════════════════════
 * SECTION 5 — CVODE / SPGMR / kernel tuning
 * ═════════════════════════════════════════════════════════════════════ */

#ifndef T_TOTAL
#define T_TOTAL 1000.0
#endif
#ifndef MAX_BDF_ORDER
#define MAX_BDF_ORDER 5
#endif
#ifndef RTOL_VAL
#define RTOL_VAL 1.0e-4
#endif
#ifndef ATOL_VAL
#define ATOL_VAL 1.0e-4
#endif
#ifndef KRYLOV_DIM
#define KRYLOV_DIM 5
#endif

/* ── CUDA block dimensions ───────────────────────────────────────────
 *
 * In EXEC_YMSK mode the per-cell kernels launch as a 2D grid of size
 *     ceil(ng / BLOCK_X) x ceil(ny / BLOCK_Y)
 * BLOCK_X * BLOCK_Y must be ≤ 1024.  Default 16x8 = 128 threads/block.
 *
 * In EXEC_COMPACT mode the per-cell kernels launch as a 1D grid of size
 *     ceil(n_active / BLOCK_SIZE)
 * BLOCK_SIZE must be ≤ 1024.  Default 256.
 *
 * Both sets of macros are defined here so a variant can read whichever
 * applies without #ifdef'ing on EXECUTION_MODEL. */
#ifndef BLOCK_X
#define BLOCK_X 16
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 8
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

/* Gram-Schmidt orthogonalization in SPGMR. */
#define GS_KIND_AUTO 0
#define GS_KIND_CGS  1
#define GS_KIND_MGS  2

#ifndef GS_KIND
#define GS_KIND GS_KIND_AUTO
#endif


/* ═════════════════════════════════════════════════════════════════════
 * SECTION 6 — Output / I/O schedule
 * ═════════════════════════════════════════════════════════════════════ */

#ifndef ENABLE_OUTPUT
#define ENABLE_OUTPUT 0
#endif
#ifndef EARLY_SAVE_UNTIL
#define EARLY_SAVE_UNTIL 80.0
#endif
#ifndef EARLY_SAVE_EVERY
#define EARLY_SAVE_EVERY 5
#endif
#ifndef LATE_SAVE_EVERY
#define LATE_SAVE_EVERY  100
#endif
#ifndef WRITE_FINAL_STATE
#define WRITE_FINAL_STATE 1
#endif


/* ═════════════════════════════════════════════════════════════════════
 * SECTION 7 — Derived constants (do not edit)
 * ═════════════════════════════════════════════════════════════════════ */

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

#if BLOCK_SIZE > 1024
#  error "BLOCK_SIZE exceeds the CUDA per-block thread limit (1024)."
#endif

#if (GEOMETRY_KIND == GEOMETRY_POLYCRYSTAL) && (NUM_GRAINS < 1)
#  error "GEOMETRY_POLYCRYSTAL requires NUM_GRAINS >= 1."
#endif

#if (GEOMETRY_KIND == GEOMETRY_RING) && \
    ((INNER_W_FRAC_OF_OUTER >= 1.0) || (INNER_H_FRAC_OF_OUTER >= 1.0))
#  error "Ring geometry requires inner < outer (INNER_*_FRAC_OF_OUTER < 1.0)."
#endif

#if (GEOMETRY_KIND == GEOMETRY_ELLIPSE) && \
    ((ACTIVE_RX_FRAC <= 0.0) || (ACTIVE_RY_FRAC <= 0.0))
#  error "Ellipse geometry requires ACTIVE_RX_FRAC > 0 and ACTIVE_RY_FRAC > 0."
#endif

#if (EXECUTION_MODEL != EXEC_YMSK) && (EXECUTION_MODEL != EXEC_COMPACT)
#  error "EXECUTION_MODEL must be EXEC_YMSK or EXEC_COMPACT."
#endif


/* ═════════════════════════════════════════════════════════════════════
 * SECTION 9 — UserData layouts (REFERENCE — for the .cu source files)
 *
 * The struct layout depends on EXECUTION_MODEL.  Three files mirror
 * the same struct: <main>.cu (UserData), jtv.cu (JtvUserData), and
 * precond.cu (PcUserData).  All three MUST match byte-for-byte.
 *
 * ──── EXEC_YMSK layout (i1..i5) ── 88 bytes ─────────────────────────
 *   offset 0  : PrecondData *pd
 *   offset 8  : DemagData   *demag
 *   offset 16 : sunrealtype *d_hdmag       (3*ncell)
 *   offset 24 : sunrealtype *d_ymsk        (3*ncell, 1 active / 0 hole)
 *   offset 32 : int nx, ny, ng, ncell, neq (5*4 = 20 B)
 *   offset 52 : 4 B padding
 *   offset 56 : double nxx0, nyy0, nzz0
 *
 * ──── EXEC_COMPACT layout (i6) ── 96 bytes ──────────────────────────
 *   offset 0  : PrecondData *pd
 *   offset 8  : DemagData   *demag
 *   offset 16 : sunrealtype *d_hdmag       (3*ncell)
 *   offset 24 : int         *d_active_ids    (n_active)
 *   offset 32 : int         *d_inactive_ids  (n_inactive)
 *   offset 40 : int nx, ny, ng, ncell, neq (5*4 = 20 B)
 *   offset 60 : int n_active, n_inactive   (2*4 = 8 B)
 *   offset 68 : 4 B padding
 *   offset 72 : double nxx0, nyy0, nzz0
 *
 * If you EXTEND either layout (e.g. add a new pointer field), you
 * MUST add the same field at the same offset in jtv.cu's JtvUserData
 * and precond.cu's PcUserData.  Build will compile fine without it
 * (the structs are all named differently) but you'll get silent
 * memory corruption at runtime.
 * ═════════════════════════════════════════════════════════════════════ */

#endif /* SIM_CONFIG_H */
