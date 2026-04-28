# =============================================================================
# params.mk  —  v2: covers 2d_i1 ... 2d_i6.
#
# Makefile fragment exposing every sim_config.h knob as a -D override.
#
# Usage:  add `include params.mk` near the top of your variant's Makefile
#         (replacing the per-variant knob block). All defaults below match
#         sim_config.h. Any value can be overridden on the command line:
#
#             make NX_VAL=3072 NY_VAL=1024 DEMAG_STRENGTH=2.0
#             make GEOMETRY_KIND=2 OUTER_W_FRAC=0.6 INNER_W_FRAC_OF_OUTER=0.4
#             make GEOMETRY_KIND=5 ACTIVE_RX_FRAC=0.3 ACTIVE_RY_FRAC=0.2
#             make EXECUTION_MODEL=1 BLOCK_SIZE=512  # i6-style compact, 1D launch
#
# At the end this file appends ALL -D flags to NVCC_FLAGS.
#
# Numeric integer codes (must match sim_config.h):
#
#   EXECUTION_MODEL :  0=YMSK (i1..i5),  1=COMPACT (i6)
#   GEOMETRY_KIND   :  0=BULK, 1=HOLE_SQUARE, 2=RING, 3=POLYCRYSTAL,
#                      4=CUSTOM, 5=ELLIPSE
#   IC_KIND         :  0=UNIFORM, 1=HEAD_ON_STRIPES, 2=TWO_DOMAIN,
#                      3=GRAIN_BUMPS, 4=CUSTOM
#   ANISO_KIND      :  0=LINEAR, 1=CUBIC
#   GS_KIND         :  0=AUTO, 1=CGS, 2=MGS
# =============================================================================

# ── Section 0: discretization ────────────────────────────────────────
NX_VAL                 ?= 1536
NY_VAL                 ?= 512

# ── Section 0.5: execution model (NEW — picks ymsk vs compact) ────
EXECUTION_MODEL        ?= 0

# ── Section 1: geometry selector ─────────────────────────────────────
GEOMETRY_KIND          ?= 1

# Square-hole knobs (GEOMETRY_KIND=1)
HOLE_CENTER_X_FRAC     ?= 0.50
HOLE_CENTER_Y_FRAC     ?= 0.50
HOLE_RADIUS_FRAC_Y     ?= 0.22

# Ring knobs (GEOMETRY_KIND=2)
OUTER_W_FRAC           ?= 0.5
OUTER_H_FRAC           ?= 0.5
INNER_W_FRAC_OF_OUTER  ?= 0.5
INNER_H_FRAC_OF_OUTER  ?= 0.5

# Polycrystal knobs (GEOMETRY_KIND=3)
NUM_GRAINS             ?= 72
DEAD_GRAIN_FRAC        ?= 0.16
HOLE_SEED              ?= 20251104
MASK_EPS_CELLS         ?= 2.2

# Ellipse knobs (GEOMETRY_KIND=5, NEW for i6)
# rx = ACTIVE_RX_FRAC * ng,  ry = ACTIVE_RY_FRAC * ny
# rx == ry on a square ng x ny grid → circle.
ACTIVE_RX_FRAC         ?= 0.25
ACTIVE_RY_FRAC         ?= 0.25

# ── Section 2: initial condition ─────────────────────────────────────
IC_KIND                ?= 0

# Uniform IC
INIT_MX                ?= 1.0
INIT_MY                ?= -0.0175
INIT_MZ                ?= 0.0

# Head-on stripes (i1)
STRIPE_LEFT_FRAC       ?= 0.25
STRIPE_RIGHT_FRAC      ?= 0.75
INIT_RANDOM_EPS        ?= 0.01
INIT_RANDOM_SEED       ?= 12345

# Two-domain (i4)
TWO_DOMAIN_SPLIT_FRAC  ?= 0.875
TWO_DOMAIN_TAIL_MY     ?= 0.0175

# Grain bumps (i3)
GRAIN_Z_BIAS           ?= 1.6
IC_CORE_MZ             ?= 0.95

# ── Section 3: physics ───────────────────────────────────────────────
PHYS_C_CHG             ?= 1.0
PHYS_C_ALPHA           ?= 0.2
PHYS_C_CHE             ?= 50.0
PHYS_C_CHK             ?= 1.0
PHYS_C_CHA             ?= 0.0
ANISO_KIND             ?= 1

PHYS_MSK_X             ?= 1.0
PHYS_MSK_Y             ?= 0.0
PHYS_MSK_Z             ?= 0.0

PHYS_C_CHB             ?= 0.3
PHYS_NSK_X             ?= 1.0
PHYS_NSK_Y             ?= 0.0
PHYS_NSK_Z             ?= 0.0

HAPP_ENABLE            ?= 0
HAPP_X                 ?= -0.2
HAPP_Y                 ?= 0.0
HAPP_Z                 ?= 0.0

# ── Section 4: demag ─────────────────────────────────────────────────
DEMAG_STRENGTH         ?= 4.0
DEMAG_THICK            ?= 1.0
DEMAG_WINDOWED         ?= 0

# ── Section 5: solver / kernel ───────────────────────────────────────
T_TOTAL                ?= 1000.0
MAX_BDF_ORDER          ?= 5
RTOL_VAL               ?= 1.0e-4
ATOL_VAL               ?= 1.0e-4
KRYLOV_DIM             ?= 5

# Block dims:
#  - YMSK mode    uses BLOCK_X * BLOCK_Y (2D launches)
#  - COMPACT mode uses BLOCK_SIZE        (1D launches over n_active)
BLOCK_X                ?= 16
BLOCK_Y                ?= 8
BLOCK_SIZE             ?= 256

GS_KIND                ?= 0

# ── Section 6: output ────────────────────────────────────────────────
ENABLE_OUTPUT          ?= 0
EARLY_SAVE_UNTIL       ?= 80.0
EARLY_SAVE_EVERY       ?= 5
LATE_SAVE_EVERY        ?= 100
WRITE_FINAL_STATE      ?= 1

# =============================================================================
# Compose every knob into NVCC_FLAGS.
# Append to your existing NVCC_FLAGS — does not clobber.
# =============================================================================
SIM_CONFIG_DEFINES := \
    -DNX_VAL=$(NX_VAL) -DNY_VAL=$(NY_VAL) \
    -DEXECUTION_MODEL=$(EXECUTION_MODEL) \
    -DGEOMETRY_KIND=$(GEOMETRY_KIND) \
    -DHOLE_CENTER_X_FRAC=$(HOLE_CENTER_X_FRAC) \
    -DHOLE_CENTER_Y_FRAC=$(HOLE_CENTER_Y_FRAC) \
    -DHOLE_RADIUS_FRAC_Y=$(HOLE_RADIUS_FRAC_Y) \
    -DOUTER_W_FRAC=$(OUTER_W_FRAC) \
    -DOUTER_H_FRAC=$(OUTER_H_FRAC) \
    -DINNER_W_FRAC_OF_OUTER=$(INNER_W_FRAC_OF_OUTER) \
    -DINNER_H_FRAC_OF_OUTER=$(INNER_H_FRAC_OF_OUTER) \
    -DNUM_GRAINS=$(NUM_GRAINS) \
    -DDEAD_GRAIN_FRAC=$(DEAD_GRAIN_FRAC) \
    -DHOLE_SEED=$(HOLE_SEED) \
    -DMASK_EPS_CELLS=$(MASK_EPS_CELLS) \
    -DACTIVE_RX_FRAC=$(ACTIVE_RX_FRAC) \
    -DACTIVE_RY_FRAC=$(ACTIVE_RY_FRAC) \
    -DIC_KIND=$(IC_KIND) \
    -DINIT_MX=$(INIT_MX) -DINIT_MY=$(INIT_MY) -DINIT_MZ=$(INIT_MZ) \
    -DSTRIPE_LEFT_FRAC=$(STRIPE_LEFT_FRAC) \
    -DSTRIPE_RIGHT_FRAC=$(STRIPE_RIGHT_FRAC) \
    -DINIT_RANDOM_EPS=$(INIT_RANDOM_EPS) \
    -DINIT_RANDOM_SEED=$(INIT_RANDOM_SEED) \
    -DTWO_DOMAIN_SPLIT_FRAC=$(TWO_DOMAIN_SPLIT_FRAC) \
    -DTWO_DOMAIN_TAIL_MY=$(TWO_DOMAIN_TAIL_MY) \
    -DGRAIN_Z_BIAS=$(GRAIN_Z_BIAS) -DIC_CORE_MZ=$(IC_CORE_MZ) \
    -DPHYS_C_CHG=$(PHYS_C_CHG) -DPHYS_C_ALPHA=$(PHYS_C_ALPHA) \
    -DPHYS_C_CHE=$(PHYS_C_CHE) \
    -DPHYS_C_CHK=$(PHYS_C_CHK) -DPHYS_C_CHA=$(PHYS_C_CHA) \
    -DANISO_KIND=$(ANISO_KIND) \
    -DPHYS_MSK_X=$(PHYS_MSK_X) -DPHYS_MSK_Y=$(PHYS_MSK_Y) -DPHYS_MSK_Z=$(PHYS_MSK_Z) \
    -DPHYS_C_CHB=$(PHYS_C_CHB) \
    -DPHYS_NSK_X=$(PHYS_NSK_X) -DPHYS_NSK_Y=$(PHYS_NSK_Y) -DPHYS_NSK_Z=$(PHYS_NSK_Z) \
    -DHAPP_ENABLE=$(HAPP_ENABLE) \
    -DHAPP_X=$(HAPP_X) -DHAPP_Y=$(HAPP_Y) -DHAPP_Z=$(HAPP_Z) \
    -DDEMAG_STRENGTH=$(DEMAG_STRENGTH) -DDEMAG_THICK=$(DEMAG_THICK) \
    -DDEMAG_WINDOWED=$(DEMAG_WINDOWED) \
    -DT_TOTAL=$(T_TOTAL) -DMAX_BDF_ORDER=$(MAX_BDF_ORDER) \
    -DRTOL_VAL=$(RTOL_VAL) -DATOL_VAL=$(ATOL_VAL) \
    -DKRYLOV_DIM=$(KRYLOV_DIM) \
    -DBLOCK_X=$(BLOCK_X) -DBLOCK_Y=$(BLOCK_Y) -DBLOCK_SIZE=$(BLOCK_SIZE) \
    -DGS_KIND=$(GS_KIND) \
    -DENABLE_OUTPUT=$(ENABLE_OUTPUT) \
    -DEARLY_SAVE_UNTIL=$(EARLY_SAVE_UNTIL) \
    -DEARLY_SAVE_EVERY=$(EARLY_SAVE_EVERY) \
    -DLATE_SAVE_EVERY=$(LATE_SAVE_EVERY) \
    -DWRITE_FINAL_STATE=$(WRITE_FINAL_STATE)

NVCC_FLAGS += $(SIM_CONFIG_DEFINES)

.PHONY: show-config
show-config:
	@echo "─── Discretization ───"
	@echo "NX_VAL=$(NX_VAL)  NY_VAL=$(NY_VAL)"
	@echo "─── Execution model ───"
	@if [ "$(EXECUTION_MODEL)" = "0" ]; then \
	  echo "  EXEC_YMSK   (i1..i5): full-grid 2D launches, output*=ymsk; BLOCK=$(BLOCK_X)x$(BLOCK_Y)"; \
	else \
	  echo "  EXEC_COMPACT (i6)  : 1D launches over n_active; BLOCK_SIZE=$(BLOCK_SIZE)"; \
	fi
	@echo "─── Geometry (KIND=$(GEOMETRY_KIND)) ───"
	@echo "  square hole:  centre=($(HOLE_CENTER_X_FRAC),$(HOLE_CENTER_Y_FRAC))  half-side=$(HOLE_RADIUS_FRAC_Y)"
	@echo "  ring:         outer=($(OUTER_W_FRAC) x $(OUTER_H_FRAC))  inner=($(INNER_W_FRAC_OF_OUTER) x $(INNER_H_FRAC_OF_OUTER))"
	@echo "  polycrystal:  ngrains=$(NUM_GRAINS)  dead=$(DEAD_GRAIN_FRAC)  seed=$(HOLE_SEED)  eps=$(MASK_EPS_CELLS)"
	@echo "  ellipse:      rx=$(ACTIVE_RX_FRAC)*ng  ry=$(ACTIVE_RY_FRAC)*ny"
	@echo "─── IC (KIND=$(IC_KIND)) ───"
	@echo "  uniform:        m=($(INIT_MX),$(INIT_MY),$(INIT_MZ))"
	@echo "  head-on stripes: q=[$(STRIPE_LEFT_FRAC),$(STRIPE_RIGHT_FRAC)]  eps=$(INIT_RANDOM_EPS)  seed=$(INIT_RANDOM_SEED)"
	@echo "  two-domain:     split=$(TWO_DOMAIN_SPLIT_FRAC)  tail my=$(TWO_DOMAIN_TAIL_MY)"
	@echo "─── Physics ───"
	@echo "  exchange  c_che=$(PHYS_C_CHE)"
	@echo "  aniso     c_chk=$(PHYS_C_CHK) c_cha=$(PHYS_C_CHA)  axis=($(PHYS_MSK_X),$(PHYS_MSK_Y),$(PHYS_MSK_Z))  kind=$(ANISO_KIND)"
	@echo "  DMI       c_chb=$(PHYS_C_CHB)  dir=($(PHYS_NSK_X),$(PHYS_NSK_Y),$(PHYS_NSK_Z))"
	@echo "  damping   c_alpha=$(PHYS_C_ALPHA)  c_chg=$(PHYS_C_CHG)"
	@echo "  applied H = (en=$(HAPP_ENABLE)) ($(HAPP_X),$(HAPP_Y),$(HAPP_Z))"
	@echo "─── Demag ───"
	@echo "  strength=$(DEMAG_STRENGTH)  thick=$(DEMAG_THICK)  windowed=$(DEMAG_WINDOWED)"
	@echo "─── Solver ───"
	@echo "  T_TOTAL=$(T_TOTAL)  RTOL=$(RTOL_VAL)  ATOL=$(ATOL_VAL)"
	@echo "  KRYLOV_DIM=$(KRYLOV_DIM)  BDF=$(MAX_BDF_ORDER)  GS_KIND=$(GS_KIND)"
	@echo "─── Output ───"
	@echo "  enable=$(ENABLE_OUTPUT)  early<=$(EARLY_SAVE_UNTIL) every $(EARLY_SAVE_EVERY)  late every $(LATE_SAVE_EVERY)"
	@echo "  final-state dump=$(WRITE_FINAL_STATE)"
