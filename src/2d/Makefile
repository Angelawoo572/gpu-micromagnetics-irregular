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

# ---- Build mode ----
BUILD ?= release
PRINT ?= 0

BLOCK_X ?= 16
BLOCK_Y ?= 8

RTOL_VAL ?= 1.0e-4
ATOL_VAL ?= 1.0e-4

ifeq ($(BUILD),debug)
  NVCC_FLAGS := $(CUDA_ARCH) -G -g -lineinfo -Xcompiler -O0
else
  NVCC_FLAGS := $(CUDA_ARCH) -O3 -lineinfo --use_fast_math
endif

NVCC_FLAGS += -DENABLE_OUTPUT=$(PRINT) \
              -DBLOCK_X=$(BLOCK_X) -DBLOCK_Y=$(BLOCK_Y) \
              -DRTOL_VAL=$(RTOL_VAL) -DATOL_VAL=$(ATOL_VAL)

TARGET := 2d
SRC := 2d.cu
ARGS ?=

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
        sweep-block sweep-tol \
        profile profile-light profile-ext stats \
        export-csv export-sqlite ui compute clean reallyclean reports

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
	  echo "Testing RTOL_VAL=$$tol ATOL_VAL=$$tol BLOCK_X=$(BLOCK_X) BLOCK_Y=$(BLOCK_Y)"; \
	  $(MAKE) clean >/dev/null 2>&1; \
	  $(MAKE) BUILD=release PRINT=0 BLOCK_X=$(BLOCK_X) BLOCK_Y=$(BLOCK_Y) RTOL_VAL=$$tol ATOL_VAL=$$tol >/dev/null && \
	  LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS) | tee run_tol_$${tol}.txt; \
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
	@[ -f "$(PROFILE_OUT).nsys-rep" ] || (echo "ERROR: $(PROFILE_OUT).nsys-rep not found. Run 'make profile' first."; exit 1)
	nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum $(PROFILE_OUT).nsys-rep > nsys_report.txt
	@echo "Wrote: nsys_report.txt"

# ====== Export ======
export-csv:
	@[ -f "$(PROFILE_OUT).nsys-rep" ] || (echo "ERROR: $(PROFILE_OUT).nsys-rep not found. Run 'make profile' first."; exit 1)
	nsys export --type csv --output ./csv_out $(PROFILE_OUT).nsys-rep
	@echo "CSV out -> ./csv_out/"

export-sqlite:
	@[ -f "$(PROFILE_OUT).nsys-rep" ] || (echo "ERROR: $(PROFILE_OUT).nsys-rep not found. Run 'make profile' first."; exit 1)
	nsys export --sqlite $(PROFILE_OUT).sqlite $(PROFILE_OUT).nsys-rep
	@echo "SQLite -> $(PROFILE_OUT).sqlite"

# ====== GUI ======
ui:
	@[ -f "$(PROFILE_OUT).nsys-rep" ] || (echo "ERROR: $(PROFILE_OUT).nsys-rep not found. Run 'make profile' first."; exit 1)
	$(NSYS_UI) $(PROFILE_OUT).nsys-rep

# ====== Nsight Compute ======
compute: $(TARGET)
	@echo "Running Nsight Compute..."
	@echo "If this fails with ERR_NVGPUCTRPERM, the machine lacks permission for GPU performance counters."
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 \
	$(NCU) $(NCU_FLAGS) ./$(TARGET) $(ARGS) || \
	(echo ""; \
	 echo "Nsight Compute failed."; \
	 echo "Likely causes:"; \
	 echo "  1) no permission for GPU performance counters"; \
	 echo "  2) driver / ncu version mismatch"; \
	 exit 1)

clean:
	rm -f $(TARGET)

reallyclean: clean
	rm -f $(PROFILE_OUT).nsys-rep $(PROFILE_OUT).qdrep $(PROFILE_OUT).sqlite ncu_$(TARGET).ncu-rep
	rm -rf csv_out
	rm -f output.txt $(TARGET)

reports:
	$(NSYS) stats --help-reports || true