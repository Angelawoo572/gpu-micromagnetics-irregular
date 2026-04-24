# ====== Build / Link paths ======
SUNDIALS_DIR := /usr/local/sundials
INCLUDE_FLAGS := -I$(SUNDIALS_DIR)/include
LIBRARY_FLAGS := -L$(SUNDIALS_DIR)/lib64

MATHDX_BASE := /opt/nvidia/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06
INCLUDE_FLAGS += -I$(MATHDX_BASE)/include \
                 -I$(MATHDX_BASE)/external/cutlass/include

LDLIBS := -lsundials_cvode -lsundials_nveccuda \
          -lsundials_sunlinsolcusolversp -lsundials_sunmatrixcusparse \
          -lsundials_core -lcusolver -lcusparse -lcudart -lcudadevrt \
          -lm -lcufft

NVCC := nvcc
CUDA_ARCH ?= -gencode arch=compute_89,code=sm_89

# ====== Build mode ======
BUILD ?= release
PRINT ?= 0

# ====== Solver / kernel params ======
BLOCK_X ?= 16
BLOCK_Y ?= 8

RTOL_VAL ?= 1.0e-4
ATOL_VAL ?= 1.0e-4
T_TOTAL  ?= 1000.0

# ====== Periodic smooth-texture params ======
CIRCLE_CENTER_X_FRAC ?= 0.50
CIRCLE_CENTER_Y_FRAC ?= 0.50
CIRCLE_RADIUS_FRAC_Y ?= 0.22

TEXTURE_CORE_MZ    ?= -0.998
TEXTURE_OUTER_MZ   ?= 0.998
TEXTURE_WIDTH_FRAC ?= 0.35
TEXTURE_EPS        ?= 1.0e-12

# ====== FFT Demagnetization (professor's Newell calt/ctt) ======
# DEMAG_STRENGTH = 0.0  → disabled (exchange-only, original behaviour)
# DEMAG_STRENGTH = 1.0  → full dipolar demag via cuFFT Z2Z
# DEMAG_THICK           → cell thickness in z (in cell-spacing units)
DEMAG_STRENGTH ?= 1.0
DEMAG_THICK    ?= 1.0

# ====== Output schedule ======
# EARLY_SAVE_UNTIL ?= 80.0
# EARLY_SAVE_EVERY ?= 5
# LATE_SAVE_EVERY  ?= 100
EARLY_SAVE_UNTIL ?= 0.0
EARLY_SAVE_EVERY ?= 100
LATE_SAVE_EVERY  ?= 100

ifeq ($(BUILD),debug)
  NVCC_FLAGS := $(CUDA_ARCH) -G -g -lineinfo -Xcompiler -O0
else
  NVCC_FLAGS := $(CUDA_ARCH) -O3 -lineinfo --use_fast_math
endif

NVCC_FLAGS += -DENABLE_OUTPUT=$(PRINT) \
              -DBLOCK_X=$(BLOCK_X) -DBLOCK_Y=$(BLOCK_Y) \
              -DRTOL_VAL=$(RTOL_VAL) -DATOL_VAL=$(ATOL_VAL) \
              -DT_TOTAL=$(T_TOTAL) \
              -DCIRCLE_CENTER_X_FRAC=$(CIRCLE_CENTER_X_FRAC) \
              -DCIRCLE_CENTER_Y_FRAC=$(CIRCLE_CENTER_Y_FRAC) \
              -DCIRCLE_RADIUS_FRAC_Y=$(CIRCLE_RADIUS_FRAC_Y) \
              -DTEXTURE_CORE_MZ=$(TEXTURE_CORE_MZ) \
              -DTEXTURE_OUTER_MZ=$(TEXTURE_OUTER_MZ) \
              -DTEXTURE_WIDTH_FRAC=$(TEXTURE_WIDTH_FRAC) \
              -DTEXTURE_EPS=$(TEXTURE_EPS) \
              -DEARLY_SAVE_UNTIL=$(EARLY_SAVE_UNTIL) \
              -DEARLY_SAVE_EVERY=$(EARLY_SAVE_EVERY) \
              -DLATE_SAVE_EVERY=$(LATE_SAVE_EVERY) \
              -DDEMAG_STRENGTH=$(DEMAG_STRENGTH) \
              -DDEMAG_THICK=$(DEMAG_THICK)

# ====== Sources & objects ======
TARGET      := 2d_fft
SRC         := 2d_fft.cu
FUSED_SRC   := deferred_nvector.cu
PRECOND_SRC := precond.cu
JTV_SRC     := jtv.cu
DEMAG_SRC   := demag_fft.cu

FUSED_OBJ   := deferred_nvector.o
PRECOND_OBJ := precond.o
JTV_OBJ     := jtv.o
DEMAG_OBJ   := demag_fft.o

OBJS := $(FUSED_OBJ) $(PRECOND_OBJ) $(JTV_OBJ) $(DEMAG_OBJ)
ARGS ?=

# ====== Nsight tools ======
NSYS ?= /opt/nvidia/nsight-systems/2025.1.3/bin/nsys
NCU  ?= /opt/nvidia/nsight-compute/2025.3.1/ncu
NSYS_UI ?= /usr/local/cuda/bin/nsight-sys

PROFILE_OUT ?= profile_$(TARGET)

NSYS_FLAGS_LIGHT := profile --trace=cuda -o $(PROFILE_OUT)
NSYS_FLAGS_STD   := profile --trace=cuda --cuda-memory-usage=true -o $(PROFILE_OUT)
NSYS_FLAGS_EXT   := profile --trace=cuda --cuda-memory-usage=true \
                    --cuda-um-cpu-page-faults=true -o $(PROFILE_OUT)

NCU_FLAGS ?= --set roofline -o ncu_$(TARGET)

.PHONY: all run run-debug build-debug build-release \
        run-print run-noprint run-demag run-no-demag \
        sweep-block sweep-tol sweep-circle sweep-demag \
        profile profile-light profile-ext stats \
        export-csv export-sqlite ui compute \
        show-config clean reallyclean reports

# ====== Build rules ======
all: $(TARGET)

$(FUSED_OBJ): $(FUSED_SRC) deferred_nvector.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math \
	    $(INCLUDE_FLAGS) -c $< -o $@

$(PRECOND_OBJ): $(PRECOND_SRC) precond.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math \
	    $(INCLUDE_FLAGS) -c $< -o $@

$(JTV_OBJ): $(JTV_SRC) jtv.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math \
	    $(INCLUDE_FLAGS) -c $< -o $@

# Professor's Newell tensor (calt/ctt) + cuFFT Z2Z convolution
$(DEMAG_OBJ): $(DEMAG_SRC) demag_fft.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math \
	    $(INCLUDE_FLAGS) -c $< -o $@

$(TARGET): $(SRC) $(OBJS) deferred_nvector.h precond.h jtv.h demag_fft.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -o $@ $< $(OBJS) \
	    $(LIBRARY_FLAGS) $(LDLIBS)

build-release:
	$(MAKE) BUILD=release PRINT=$(PRINT)

build-debug:
	$(MAKE) BUILD=debug PRINT=$(PRINT)

# ====== Run targets ======
run: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS)

run-debug: $(TARGET)
	CUDA_LAUNCH_BLOCKING=1 LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS)

run-print:
	$(MAKE) clean
	$(MAKE) PRINT=1
	$(MAKE) run PRINT=1

run-noprint:
	$(MAKE) clean
	$(MAKE) PRINT=0
	$(MAKE) run PRINT=0

# Full demag (professor's Newell tensor)
run-demag:
	$(MAKE) clean
	$(MAKE) DEMAG_STRENGTH=1.0 DEMAG_THICK=1.0
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS)

# Exchange-only baseline (demag disabled, recovers original behaviour)
run-no-demag:
	$(MAKE) clean
	$(MAKE) DEMAG_STRENGTH=0.0
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS)

# ====== Parameter sweeps ======
sweep-block:
	@for bx in 32 16; do \
	  for by in 4 8 16; do \
	    threads=`expr $$bx \* $$by`; \
	    if [ $$threads -le 1024 ]; then \
	      echo "========================================"; \
	      echo "Testing BLOCK_X=$$bx BLOCK_Y=$$by"; \
	      $(MAKE) clean >/dev/null 2>&1; \
	      $(MAKE) BUILD=release PRINT=0 BLOCK_X=$$bx BLOCK_Y=$$by >/dev/null && \
	      LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS) | tee run_bx$${bx}_by$${by}.txt; \
	    fi; \
	  done; \
	done

sweep-tol:
	@for tol in 1.0e-4 1.0e-5 1.0e-6; do \
	  echo "========================================"; \
	  echo "Testing RTOL_VAL=$$tol ATOL_VAL=$$tol"; \
	  $(MAKE) clean >/dev/null 2>&1; \
	  $(MAKE) BUILD=release PRINT=0 RTOL_VAL=$$tol ATOL_VAL=$$tol >/dev/null && \
	  LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS) | tee run_tol_$${tol}.txt; \
	done

sweep-circle:
	@for r in 0.12 0.18 0.22 0.28 0.34; do \
	  echo "========================================"; \
	  echo "Testing CIRCLE_RADIUS_FRAC_Y=$$r"; \
	  $(MAKE) clean >/dev/null 2>&1; \
	  $(MAKE) BUILD=release PRINT=0 \
	    CIRCLE_RADIUS_FRAC_Y=$$r >/dev/null && \
	  LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS) | tee run_circle_r_$${r}.txt; \
	done

sweep-demag:
	@for s in 0.0 0.01 0.1 0.5 1.0; do \
	  echo "========================================"; \
	  echo "Testing DEMAG_STRENGTH=$$s"; \
	  $(MAKE) clean >/dev/null 2>&1; \
	  $(MAKE) BUILD=release PRINT=0 DEMAG_STRENGTH=$$s DEMAG_THICK=1.0 >/dev/null && \
	  LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS) | tee run_demag_$${s}.txt; \
	done

# ====== Nsight Systems ======
profile: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 \
	$(NSYS) $(NSYS_FLAGS_STD) ./$(TARGET) $(ARGS)
	@echo "Generated: $(PROFILE_OUT).nsys-rep"

profile-light: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 \
	$(NSYS) $(NSYS_FLAGS_LIGHT) ./$(TARGET) $(ARGS)
	@echo "Generated: $(PROFILE_OUT).nsys-rep"

profile-ext: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 \
	$(NSYS) $(NSYS_FLAGS_EXT) ./$(TARGET) $(ARGS)
	@echo "Generated: $(PROFILE_OUT).nsys-rep"

stats:
	@[ -f "$(PROFILE_OUT).nsys-rep" ] || (echo "ERROR: run make profile first."; exit 1)
	nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
	    $(PROFILE_OUT).nsys-rep > nsys_report.txt
	@echo "Wrote: nsys_report.txt"

export-csv:
	@[ -f "$(PROFILE_OUT).nsys-rep" ] || (echo "ERROR: run make profile first."; exit 1)
	nsys export --type csv --output ./csv_out $(PROFILE_OUT).nsys-rep

export-sqlite:
	@[ -f "$(PROFILE_OUT).nsys-rep" ] || (echo "ERROR: run make profile first."; exit 1)
	nsys export --sqlite $(PROFILE_OUT).sqlite $(PROFILE_OUT).nsys-rep

ui:
	@[ -f "$(PROFILE_OUT).nsys-rep" ] || (echo "ERROR: run make profile first."; exit 1)
	$(NSYS_UI) $(PROFILE_OUT).nsys-rep

compute: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 \
	$(NCU) $(NCU_FLAGS) ./$(TARGET) $(ARGS) || \
	(echo "Nsight Compute failed (check GPU counter permissions)"; exit 1)

# ====== Show config ======
show-config:
	@echo "TARGET         = $(TARGET)"
	@echo "DEMAG_STRENGTH = $(DEMAG_STRENGTH)  (0=off, 1=full Newell tensor)"
	@echo "DEMAG_THICK    = $(DEMAG_THICK)"
	@echo "RTOL_VAL       = $(RTOL_VAL)"
	@echo "ATOL_VAL       = $(ATOL_VAL)"
	@echo "T_TOTAL        = $(T_TOTAL)"
	@echo "BLOCK_X        = $(BLOCK_X)   BLOCK_Y = $(BLOCK_Y)"
	@echo "CUDA_ARCH      = $(CUDA_ARCH)"
	@echo "PROFILE_OUT    = $(PROFILE_OUT)"


# ====== Direct (O(N²)) target — drop-in swap of demag module ======
# Uses same 2d_fft.cu but links demag_direct.o instead of demag_fft.o
# Everything else (CVODE, precond, jtv, fused_nvec) is identical.
#
# Build:
#   make 2d_direct
# Run:
#   make run-direct
# Compare timing:
#   make bench

DIRECT_TARGET  := 2d_direct
DIRECT_DEMAG_SRC := demag_direct.cu
DIRECT_DEMAG_OBJ := demag_direct.o
DIRECT_OBJS    := $(FUSED_OBJ) $(PRECOND_OBJ) $(JTV_OBJ) $(DIRECT_DEMAG_OBJ)

$(DIRECT_DEMAG_OBJ): $(DIRECT_DEMAG_SRC) demag_fft.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math \
	    $(INCLUDE_FLAGS) -c $< -o $@

# Note: SRC (2d_fft.cu) is reused — it includes demag_fft.h which both
# demag_fft.cu and demag_direct.cu implement. Only the linked .o differs.
$(DIRECT_TARGET): 2d_fft.cu $(DIRECT_OBJS) deferred_nvector.h precond.h jtv.h demag_fft.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -o $@ 2d_fft.cu $(DIRECT_OBJS) \
	    $(LIBRARY_FLAGS) $(LDLIBS)

run-direct: $(DIRECT_TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(DIRECT_TARGET) $(ARGS)

# ── Side-by-side timing benchmark ──────────────────────────────────────
# Builds both targets and runs each once, printing GPU time.
# Redirect stderr to /dev/null to suppress calt progress messages.
bench: $(TARGET) $(DIRECT_TARGET)
	@echo "========================================================"
	@echo "FFT version    (O(N log N)):"
	@echo "========================================================"
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS)
	@echo ""
	@echo "========================================================"
	@echo "Direct version (O(N²)):"
	@echo "========================================================"
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(DIRECT_TARGET) $(ARGS)

.PHONY: run-direct bench

# ====== Clean ======
clean:
	rm -f $(TARGET) $(DIRECT_TARGET)
	rm -f $(OBJS) $(DIRECT_DEMAG_OBJ)

reallyclean: clean
	rm -f $(PROFILE_OUT).nsys-rep $(PROFILE_OUT).qdrep $(PROFILE_OUT).sqlite
	rm -f ncu_$(TARGET).ncu-rep ncu_$(DIRECT_TARGET).ncu-rep
	rm -rf csv_out
	rm -f *.txt

reports:
	$(NSYS) stats --help-reports || true