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

# ====== Output schedule ======
EARLY_SAVE_UNTIL ?= 80.0
EARLY_SAVE_EVERY ?= 5
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
              -DLATE_SAVE_EVERY=$(LATE_SAVE_EVERY)

# ====== Target ======
TARGET       := 2d_p
SRC          := 2d_p.cu
FUSED_SRC    := deferred_nvector.cu
PRECOND_SRC  := precond.cu
JTV_SRC      := jtv.cu

FUSED_OBJ    := deferred_nvector.o
PRECOND_OBJ  := precond.o
JTV_OBJ      := jtv.o
OBJS         := $(FUSED_OBJ) $(PRECOND_OBJ) $(JTV_OBJ)

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
        sweep-block sweep-tol sweep-circle \
        profile profile-light profile-ext stats \
        export-csv export-sqlite ui compute \
        show-config clean reallyclean reports \
        bench-fused

# ====== Build ======
all: $(TARGET)

# compile fused NVector layer
$(FUSED_OBJ): $(FUSED_SRC) deferred_nvector.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math \
	    $(INCLUDE_FLAGS) -c $< -o $@

# compile 3x3 block Jacobi preconditioner
$(PRECOND_OBJ): $(PRECOND_SRC) precond.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math \
	    $(INCLUDE_FLAGS) -c $< -o $@

# compile analytic Jacobian-times-vector
# <<< JTV: eliminates ~20K extra f() calls per run
$(JTV_OBJ): $(JTV_SRC) jtv.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math \
	    $(INCLUDE_FLAGS) -c $< -o $@

# compile and link main binary
$(TARGET): $(SRC) $(OBJS) deferred_nvector.h precond.h jtv.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -o $@ $< $(OBJS) \
	    $(LIBRARY_FLAGS) $(LDLIBS)

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

# ====== Quick benchmark target ======
bench-fused: $(TARGET)
	@echo "=== Fused + Precond + Analytic Jv ==="
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

# ====== Stats ======
stats:
	@[ -f "$(PROFILE_OUT).nsys-rep" ] || (echo "ERROR: run make profile first."; exit 1)
	nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum $(PROFILE_OUT).nsys-rep > nsys_report.txt
	@echo "Wrote: nsys_report.txt"

# ====== Export ======
export-csv:
	@[ -f "$(PROFILE_OUT).nsys-rep" ] || (echo "ERROR: run make profile first."; exit 1)
	nsys export --type csv --output ./csv_out $(PROFILE_OUT).nsys-rep
	@echo "CSV out -> ./csv_out/"

export-sqlite:
	@[ -f "$(PROFILE_OUT).nsys-rep" ] || (echo "ERROR: run make profile first."; exit 1)
	nsys export --sqlite $(PROFILE_OUT).sqlite $(PROFILE_OUT).nsys-rep
	@echo "SQLite -> $(PROFILE_OUT).sqlite"

# ====== GUI ======
ui:
	@[ -f "$(PROFILE_OUT).nsys-rep" ] || (echo "ERROR: run make profile first."; exit 1)
	$(NSYS_UI) $(PROFILE_OUT).nsys-rep

# ====== Nsight Compute ======
compute: $(TARGET)
	@echo "Running Nsight Compute..."
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 \
	$(NCU) $(NCU_FLAGS) ./$(TARGET) $(ARGS) || \
	(echo "Nsight Compute failed (check GPU counter permissions)"; exit 1)

# ====== Show config ======
show-config:
	@echo "TARGET                = $(TARGET)"
	@echo "SRC                   = $(SRC)"
	@echo "FUSED_SRC             = $(FUSED_SRC)"
	@echo "PRECOND_SRC           = $(PRECOND_SRC)"
	@echo "JTV_SRC               = $(JTV_SRC)   (analytic Jv)"
	@echo "OBJS                  = $(OBJS)"
	@echo "BUILD                 = $(BUILD)"
	@echo "CUDA_ARCH             = $(CUDA_ARCH)"
	@echo "BLOCK_X               = $(BLOCK_X)   BLOCK_Y = $(BLOCK_Y)"
	@echo "RTOL_VAL              = $(RTOL_VAL)"
	@echo "ATOL_VAL              = $(ATOL_VAL)"
	@echo "T_TOTAL               = $(T_TOTAL)"
	@echo "PROFILE_OUT           = $(PROFILE_OUT)"

# ====== Clean ======
clean:
	rm -f $(TARGET) $(OBJS)

reallyclean: clean
	rm -f $(PROFILE_OUT).nsys-rep $(PROFILE_OUT).qdrep $(PROFILE_OUT).sqlite ncu_$(TARGET).ncu-rep
	rm -rf csv_out
	rm -f output.txt run_bx*.txt run_tol_*.txt run_circle_r_*.txt

reports:
	$(NSYS) stats --help-reports || true
