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
# i6 uses 1D compact launches (one thread per active cell), so a single
# BLOCK_SIZE replaces the i5 BLOCK_X/BLOCK_Y pair.
BLOCK_SIZE ?= 256

RTOL_VAL ?= 1.0e-4
ATOL_VAL ?= 1.0e-4
T_TOTAL  ?= 1000.0

KRYLOV_DIM ?= 5

# ====== FFT Demagnetization ======
DEMAG_STRENGTH ?= 4.0
DEMAG_THICK    ?= 1.0

# ====== i6 circular active-region geometry ======
# Centered circle of radius  ACTIVE_RADIUS_FRAC * min(ng, ny)  cells.
# Cells inside the circle are active; everything outside is hole.
ACTIVE_RADIUS_FRAC ?= 0.25

# ====== Output schedule ======
EARLY_SAVE_UNTIL ?= 100.0
EARLY_SAVE_EVERY ?= 50
LATE_SAVE_EVERY  ?= 500

ifeq ($(BUILD),debug)
  NVCC_FLAGS := $(CUDA_ARCH) -G -g -lineinfo -Xcompiler -O0
else
  NVCC_FLAGS := $(CUDA_ARCH) -O3 -lineinfo --use_fast_math
endif

NVCC_FLAGS += -DENABLE_OUTPUT=$(PRINT) \
              -DBLOCK_SIZE=$(BLOCK_SIZE) \
              -DRTOL_VAL=$(RTOL_VAL) -DATOL_VAL=$(ATOL_VAL) \
              -DT_TOTAL=$(T_TOTAL) \
              -DKRYLOV_DIM=$(KRYLOV_DIM) \
              -DEARLY_SAVE_UNTIL=$(EARLY_SAVE_UNTIL) \
              -DEARLY_SAVE_EVERY=$(EARLY_SAVE_EVERY) \
              -DLATE_SAVE_EVERY=$(LATE_SAVE_EVERY) \
              -DDEMAG_STRENGTH=$(DEMAG_STRENGTH) \
              -DDEMAG_THICK=$(DEMAG_THICK) \
              -DACTIVE_RADIUS_FRAC=$(ACTIVE_RADIUS_FRAC)

# ====== Sources & objects ======
TARGET      := 2d_i6
SRC         := 2d_i6.cu
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
        sweep-block sweep-tol sweep-demag sweep-krylov sweep-radius \
        profile profile-light profile-ext stats \
        export-csv export-sqlite ui compute \
        show-config clean reallyclean reports \
        run-direct bench

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

run-demag:
	$(MAKE) clean
	$(MAKE) DEMAG_STRENGTH=5.0 DEMAG_THICK=8.0
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS)

run-no-demag:
	$(MAKE) clean
	$(MAKE) DEMAG_STRENGTH=0.0
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS)

# ====== Parameter sweeps ======
# 1D compact launches → one BLOCK_SIZE knob.
sweep-block:
	@for bs in 128 256 512; do \
	  if [ $$bs -le 1024 ]; then \
	    echo "========================================"; \
	    echo "Testing BLOCK_SIZE=$$bs"; \
	    $(MAKE) clean >/dev/null 2>&1; \
	    $(MAKE) BUILD=release PRINT=0 BLOCK_SIZE=$$bs >/dev/null && \
	    LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS) | tee run_bs_$${bs}.txt; \
	  fi; \
	done

sweep-tol:
	@for tol in 1.0e-4 1.0e-5 1.0e-6; do \
	  echo "========================================"; \
	  echo "Testing RTOL_VAL=$$tol ATOL_VAL=$$tol"; \
	  $(MAKE) clean >/dev/null 2>&1; \
	  $(MAKE) BUILD=release PRINT=0 RTOL_VAL=$$tol ATOL_VAL=$$tol >/dev/null && \
	  LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS) | tee run_tol_$${tol}.txt; \
	done

sweep-demag:
	@for s in 0.0 0.01 0.1 0.5 1.0 2.0; do \
	  echo "========================================"; \
	  echo "Testing DEMAG_STRENGTH=$$s"; \
	  $(MAKE) clean >/dev/null 2>&1; \
	  $(MAKE) BUILD=release PRINT=0 DEMAG_STRENGTH=$$s DEMAG_THICK=1.0 >/dev/null && \
	  LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS) | tee run_demag_$${s}.txt; \
	done

sweep-krylov:
	@for kd in 0 5 10 15 20 30; do \
	  echo "========================================"; \
	  echo "Testing KRYLOV_DIM=$$kd"; \
	  $(MAKE) clean >/dev/null 2>&1; \
	  $(MAKE) BUILD=release PRINT=0 KRYLOV_DIM=$$kd >/dev/null && \
	  LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS) | tee run_krylov_$${kd}.txt; \
	done

# Sweep the circle radius fraction (active-region size).
# Smaller radius → fewer active cells, larger speedup vs full-grid.
sweep-radius:
	@for r in 0.10 0.15 0.20 0.25 0.30; do \
	  echo "========================================"; \
	  echo "Testing ACTIVE_RADIUS_FRAC=$$r"; \
	  $(MAKE) clean >/dev/null 2>&1; \
	  $(MAKE) BUILD=release PRINT=0 ACTIVE_RADIUS_FRAC=$$r >/dev/null && \
	  LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS) | tee run_radius_$${r}.txt; \
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
	@echo "TARGET                  = $(TARGET)"
	@echo "SRC                     = $(SRC)"
	@echo "DEMAG_STRENGTH          = $(DEMAG_STRENGTH)"
	@echo "DEMAG_THICK             = $(DEMAG_THICK)"
	@echo "ACTIVE_RADIUS_FRAC      = $(ACTIVE_RADIUS_FRAC)"
	@echo "KRYLOV_DIM              = $(KRYLOV_DIM)"
	@echo "RTOL_VAL                = $(RTOL_VAL)"
	@echo "ATOL_VAL                = $(ATOL_VAL)"
	@echo "T_TOTAL                 = $(T_TOTAL)"
	@echo "BLOCK_SIZE              = $(BLOCK_SIZE)"
	@echo "CUDA_ARCH               = $(CUDA_ARCH)"
	@echo "PROFILE_OUT             = $(PROFILE_OUT)"

# ====== Direct O(N^2) target ======
DIRECT_TARGET    := 2d_i6_direct
DIRECT_DEMAG_SRC := demag_direct.cu
DIRECT_DEMAG_OBJ := demag_direct.o
DIRECT_OBJS      := $(FUSED_OBJ) $(PRECOND_OBJ) $(JTV_OBJ) $(DIRECT_DEMAG_OBJ)

$(DIRECT_DEMAG_OBJ): $(DIRECT_DEMAG_SRC) demag_fft.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math \
	    $(INCLUDE_FLAGS) -c $< -o $@

$(DIRECT_TARGET): $(SRC) $(DIRECT_OBJS) deferred_nvector.h precond.h jtv.h demag_fft.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -o $@ $(SRC) $(DIRECT_OBJS) \
	    $(LIBRARY_FLAGS) $(LDLIBS)

run-direct: $(DIRECT_TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(DIRECT_TARGET) $(ARGS)

bench: $(TARGET) $(DIRECT_TARGET)
	@echo "========================================================"
	@echo "FFT version    (O(N log N)):"
	@echo "========================================================"
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS)
	@echo ""
	@echo "========================================================"
	@echo "Direct version (O(N^2)):"
	@echo "========================================================"
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(DIRECT_TARGET) $(ARGS)

# ====== Clean ======
clean:
	rm -f $(TARGET) $(DIRECT_TARGET)
	rm -f $(OBJS) $(DIRECT_DEMAG_OBJ)

reallyclean: clean
	rm -f $(PROFILE_OUT).nsys-rep $(PROFILE_OUT).qdrep $(PROFILE_OUT).sqlite
	rm -f ncu_$(TARGET).ncu-rep ncu_$(DIRECT_TARGET).ncu-rep
	rm -rf csv_out
	rm -f output.txt

reports:
	$(NSYS) stats --help-reports || true
