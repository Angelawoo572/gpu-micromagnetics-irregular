# ====== Build / Link paths ======
SUNDIALS_DIR := /usr/local/sundials
INCLUDE_FLAGS := -I$(SUNDIALS_DIR)/include
LIBRARY_FLAGS := -L$(SUNDIALS_DIR)/lib64

# cuFFTDx / MathDX
MATHDX_BASE := /opt/nvidia/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06
INCLUDE_FLAGS += -I$(MATHDX_BASE)/include \
                 -I$(MATHDX_BASE)/external/cutlass/include

LDLIBS := -lsundials_cvode -lsundials_nveccuda \
          -lsundials_sunlinsolcusolversp -lsundials_sunmatrixcusparse \
          -lsundials_core -lcusolver -lcusparse -lcudart -lcudadevrt \
          -lm -lcufft

NVCC := nvcc
CUDA_ARCH ?= -arch=sm_89

# ====== Build mode ======
BUILD ?= release
PRINT ?= 0

# ====== Solver / kernel params ======
BLOCK_X ?= 16
BLOCK_Y ?= 8

RTOL_VAL ?= 1.0e-4
ATOL_VAL ?= 1.0e-4

# ====== Growing circle params (NEW) ======
CIRCLE_CENTER_X_FRAC ?= 0.50
CIRCLE_CENTER_Y_FRAC ?= 0.50

R0_FRAC_Y ?= 0.10
R_GROWTH_FRAC_Y_PER_T ?= 0.0002
WALL_WIDTH_FRAC_Y ?= 0.06
SEED_AMP ?= 0.20

INIT_MY ?= 0.0
T_TOTAL ?= 1000.0

ifeq ($(BUILD),debug)
  NVCC_FLAGS := $(CUDA_ARCH) -G -g -lineinfo -Xcompiler -O0
else
  NVCC_FLAGS := $(CUDA_ARCH) -O3 -lineinfo --use_fast_math
endif

# ====== Pass macros ======
NVCC_FLAGS += -DENABLE_OUTPUT=$(PRINT) \
              -DBLOCK_X=$(BLOCK_X) -DBLOCK_Y=$(BLOCK_Y) \
              -DRTOL_VAL=$(RTOL_VAL) -DATOL_VAL=$(ATOL_VAL) \
              -DCIRCLE_CENTER_X_FRAC=$(CIRCLE_CENTER_X_FRAC) \
              -DCIRCLE_CENTER_Y_FRAC=$(CIRCLE_CENTER_Y_FRAC) \
              -DR0_FRAC_Y=$(R0_FRAC_Y) \
              -DR_GROWTH_FRAC_Y_PER_T=$(R_GROWTH_FRAC_Y_PER_T) \
              -DWALL_WIDTH_FRAC_Y=$(WALL_WIDTH_FRAC_Y) \
              -DSEED_AMP=$(SEED_AMP) \
              -DINIT_MY=$(INIT_MY) \
              -DT_TOTAL=$(T_TOTAL)

# ====== Target ======
TARGET := 2d_p
SRC := 2d_p.cu
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
        run-print run-noprint \
        sweep-block sweep-tol sweep-growth \
        profile profile-light profile-ext stats \
        export-csv export-sqlite ui compute \
        show-config clean reallyclean reports

# ====== Build ======
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -o $@ $< $(LIBRARY_FLAGS) $(LDLIBS)

build-release:
	$(MAKE) BUILD=release PRINT=$(PRINT)

build-debug:
	$(MAKE) BUILD=debug PRINT=$(PRINT)

# ====== Run ======
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

# ====== Parameter sweeps ======

# block tuning
sweep-block:
	@for bx in 32 16; do \
	  for by in 4 8 16; do \
	    threads=`expr $$bx \* $$by`; \
	    if [ $$threads -le 1024 ]; then \
	      echo "========================================"; \
	      echo "Testing BLOCK_X=$$bx BLOCK_Y=$$by"; \
	      $(MAKE) clean >/dev/null 2>&1; \
	      $(MAKE) BUILD=release PRINT=0 BLOCK_X=$$bx BLOCK_Y=$$by >/dev/null && \
	      LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) | tee run_bx$${bx}_by$${by}.txt; \
	    fi; \
	  done; \
	done

# tolerance sweep
sweep-tol:
	@for tol in 1.0e-4 1.0e-5 1.0e-6; do \
	  echo "========================================"; \
	  echo "Testing tol=$$tol"; \
	  $(MAKE) clean >/dev/null 2>&1; \
	  $(MAKE) BUILD=release PRINT=0 RTOL_VAL=$$tol ATOL_VAL=$$tol >/dev/null && \
	  LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) | tee run_tol_$${tol}.txt; \
	done

# ⭐ NEW: growth sweep（最重要）
sweep-growth:
	@for gr in 0.0001 0.0002 0.0004 0.0008; do \
	  echo "========================================"; \
	  echo "Testing growth=$$gr"; \
	  $(MAKE) clean >/dev/null 2>&1; \
	  $(MAKE) BUILD=release PRINT=0 \
	    R_GROWTH_FRAC_Y_PER_T=$$gr \
	    R0_FRAC_Y=$(R0_FRAC_Y) \
	    WALL_WIDTH_FRAC_Y=$(WALL_WIDTH_FRAC_Y) \
	    SEED_AMP=$(SEED_AMP) >/dev/null && \
	  LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) | tee run_growth_$${gr}.txt; \
	done

# ====== Nsight Systems ======
profile: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 \
	$(NSYS) $(NSYS_FLAGS_STD) ./$(TARGET)
	@echo "Generated: $(PROFILE_OUT).nsys-rep"

profile-light: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 \
	$(NSYS) $(NSYS_FLAGS_LIGHT) ./$(TARGET)

profile-ext: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 \
	$(NSYS) $(NSYS_FLAGS_EXT) ./$(TARGET)

# ====== Stats ======
stats:
	nsys stats --report cuda_api_sum,cuda_gpu_kern_sum $(PROFILE_OUT).nsys-rep > nsys_report.txt

# ====== GUI ======
ui:
	$(NSYS_UI) $(PROFILE_OUT).nsys-rep

# ====== Nsight Compute ======
compute: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 \
	$(NCU) $(NCU_FLAGS) ./$(TARGET)

# ====== Show config ======
show-config:
	@echo "BLOCK_X=$(BLOCK_X) BLOCK_Y=$(BLOCK_Y)"
	@echo "R0_FRAC_Y=$(R0_FRAC_Y)"
	@echo "R_GROWTH=$(R_GROWTH_FRAC_Y_PER_T)"
	@echo "WALL_WIDTH=$(WALL_WIDTH_FRAC_Y)"
	@echo "SEED_AMP=$(SEED_AMP)"
	@echo "T_TOTAL=$(T_TOTAL)"

# ====== Clean ======
clean:
	rm -f $(TARGET)

reallyclean: clean
	rm -f *.nsys-rep *.sqlite *.txt output.txt