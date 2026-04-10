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

# ====== Problem size / solver / kernel params ======
NX ?= 3000
NY ?= 1280
BLOCK_X ?= 16
BLOCK_Y ?= 8
TILE_X ?= 16
TILE_Y ?= 8
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

# ====== Ring-aware tile scheduling params ======
# REGION_MODE:
#   0 = uniform full-grid baseline
#   1 = halo-aware binary ring tile partition
REGION_MODE ?= 0
ACTIVE_RING_RADIUS_FRAC_Y ?= $(CIRCLE_RADIUS_FRAC_Y)
ACTIVE_RING_HALF_WIDTH_FRAC_Y ?= 0.08
CHEAP_MODE ?= 0
HALO_W ?= 1
TILE_SAFETY_FRAC_Y ?= 0.02

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
              -DNX=$(NX) -DNY=$(NY) \
              -DBLOCK_X=$(BLOCK_X) -DBLOCK_Y=$(BLOCK_Y) \
              -DTILE_X=$(TILE_X) -DTILE_Y=$(TILE_Y) \
              -DRTOL_VAL=$(RTOL_VAL) -DATOL_VAL=$(ATOL_VAL) \
              -DT_TOTAL=$(T_TOTAL) \
              -DCIRCLE_CENTER_X_FRAC=$(CIRCLE_CENTER_X_FRAC) \
              -DCIRCLE_CENTER_Y_FRAC=$(CIRCLE_CENTER_Y_FRAC) \
              -DCIRCLE_RADIUS_FRAC_Y=$(CIRCLE_RADIUS_FRAC_Y) \
              -DTEXTURE_CORE_MZ=$(TEXTURE_CORE_MZ) \
              -DTEXTURE_OUTER_MZ=$(TEXTURE_OUTER_MZ) \
              -DTEXTURE_WIDTH_FRAC=$(TEXTURE_WIDTH_FRAC) \
              -DTEXTURE_EPS=$(TEXTURE_EPS) \
              -DREGION_MODE=$(REGION_MODE) \
              -DACTIVE_RING_RADIUS_FRAC_Y=$(ACTIVE_RING_RADIUS_FRAC_Y) \
              -DACTIVE_RING_HALF_WIDTH_FRAC_Y=$(ACTIVE_RING_HALF_WIDTH_FRAC_Y) \
              -DCHEAP_MODE=$(CHEAP_MODE) \
              -DHALO_W=$(HALO_W) \
              -DTILE_SAFETY_FRAC_Y=$(TILE_SAFETY_FRAC_Y) \
              -DEARLY_SAVE_UNTIL=$(EARLY_SAVE_UNTIL) \
              -DEARLY_SAVE_EVERY=$(EARLY_SAVE_EVERY) \
              -DLATE_SAVE_EVERY=$(LATE_SAVE_EVERY)

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
        run-print run-noprint bench \
        sweep-block sweep-tol sweep-circle sweep-size sweep-ring \
        profile profile-short profile-light profile-ext stats \
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

bench:
	$(MAKE) clean >/dev/null 2>&1
	$(MAKE) BUILD=release PRINT=0 >/dev/null
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS) | tee bench_$(TARGET).txt

# ====== Parameter sweeps ======
sweep-block:
	@for bx in 32 16; do \
	  for by in 4 8 16; do \
	    threads=`expr $$bx \* $$by`; \
	    if [ $$threads -le 1024 ]; then \
	      echo "========================================"; \
	      echo "Testing BLOCK_X=$$bx BLOCK_Y=$$by"; \
	      $(MAKE) clean >/dev/null 2>&1; \
	      $(MAKE) BUILD=release PRINT=0 BLOCK_X=$$bx BLOCK_Y=$$by TILE_X=$$bx TILE_Y=$$by >/dev/null && \
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
	  $(MAKE) BUILD=release PRINT=0 CIRCLE_RADIUS_FRAC_Y=$$r ACTIVE_RING_RADIUS_FRAC_Y=$$r >/dev/null && \
	  LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS) | tee run_circle_r_$${r}.txt; \
	done

sweep-size:
	@for nx in 600 1024 1536; do \
	  echo "========================================"; \
	  echo "Testing NX=$$nx NY=$(NY)"; \
	  $(MAKE) clean >/dev/null 2>&1; \
	  $(MAKE) BUILD=release PRINT=0 NX=$$nx >/dev/null && \
	  LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS) | tee run_nx$${nx}_ny$(NY).txt; \
	done

sweep-ring:
	@for halfw in 0.06 0.08 0.10; do \
	  for halo in 1 2; do \
	    echo "========================================"; \
	    echo "Testing REGION_MODE=1 ACTIVE_RING_HALF_WIDTH_FRAC_Y=$$halfw HALO_W=$$halo"; \
	    $(MAKE) clean >/dev/null 2>&1; \
	    $(MAKE) BUILD=release PRINT=0 REGION_MODE=1 CHEAP_MODE=$(CHEAP_MODE) \
	      ACTIVE_RING_HALF_WIDTH_FRAC_Y=$$halfw HALO_W=$$halo >/dev/null && \
	    LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS) | tee run_ring_hw$${halfw}_halo$${halo}.txt; \
	  done; \
	done

# ====== Nsight Systems ======
profile: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 \
	$(NSYS) $(NSYS_FLAGS_STD) ./$(TARGET) $(ARGS)
	@echo "Generated: $(PROFILE_OUT).nsys-rep"

profile-short:
	$(MAKE) clean >/dev/null 2>&1
	$(MAKE) BUILD=release PRINT=0 T_TOTAL=20.0 >/dev/null && \
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
	@[ -f "$(PROFILE_OUT).nsys-rep" ] || (echo "ERROR: $(PROFILE_OUT).nsys-rep not found. Run 'make profile' or 'make profile-short' first."; exit 1)
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

# ====== Show config ======
show-config:
	@echo "TARGET                          = $(TARGET)"
	@echo "SRC                             = $(SRC)"
	@echo "BUILD                           = $(BUILD)"
	@echo "PRINT                           = $(PRINT)"
	@echo "CUDA_ARCH                       = $(CUDA_ARCH)"
	@echo "NX                              = $(NX)"
	@echo "NY                              = $(NY)"
	@echo "BLOCK_X                         = $(BLOCK_X)"
	@echo "BLOCK_Y                         = $(BLOCK_Y)"
	@echo "TILE_X                          = $(TILE_X)"
	@echo "TILE_Y                          = $(TILE_Y)"
	@echo "RTOL_VAL                        = $(RTOL_VAL)"
	@echo "ATOL_VAL                        = $(ATOL_VAL)"
	@echo "T_TOTAL                         = $(T_TOTAL)"
	@echo "REGION_MODE                     = $(REGION_MODE)"
	@echo "ACTIVE_RING_RADIUS_FRAC_Y       = $(ACTIVE_RING_RADIUS_FRAC_Y)"
	@echo "ACTIVE_RING_HALF_WIDTH_FRAC_Y   = $(ACTIVE_RING_HALF_WIDTH_FRAC_Y)"
	@echo "CHEAP_MODE                      = $(CHEAP_MODE)"
	@echo "HALO_W                          = $(HALO_W)"
	@echo "TILE_SAFETY_FRAC_Y              = $(TILE_SAFETY_FRAC_Y)"
	@echo "CIRCLE_CENTER_X_FRAC            = $(CIRCLE_CENTER_X_FRAC)"
	@echo "CIRCLE_CENTER_Y_FRAC            = $(CIRCLE_CENTER_Y_FRAC)"
	@echo "CIRCLE_RADIUS_FRAC_Y            = $(CIRCLE_RADIUS_FRAC_Y)"
	@echo "TEXTURE_CORE_MZ                 = $(TEXTURE_CORE_MZ)"
	@echo "TEXTURE_OUTER_MZ                = $(TEXTURE_OUTER_MZ)"
	@echo "TEXTURE_WIDTH_FRAC              = $(TEXTURE_WIDTH_FRAC)"
	@echo "TEXTURE_EPS                     = $(TEXTURE_EPS)"
	@echo "EARLY_SAVE_UNTIL                = $(EARLY_SAVE_UNTIL)"
	@echo "EARLY_SAVE_EVERY                = $(EARLY_SAVE_EVERY)"
	@echo "LATE_SAVE_EVERY                 = $(LATE_SAVE_EVERY)"
	@echo "PROFILE_OUT                     = $(PROFILE_OUT)"

# ====== Clean ======
clean:
	rm -f $(TARGET)

reallyclean: clean
	rm -f $(PROFILE_OUT).nsys-rep $(PROFILE_OUT).qdrep $(PROFILE_OUT).sqlite ncu_$(TARGET).ncu-rep
	rm -rf csv_out
	rm -f output.txt bench_$(TARGET).txt
	rm -f run_bx*.txt run_tol_*.txt run_circle_r_*.txt run_nx*.txt run_ring_*.txt

reports:
	$(NSYS) stats --help-reports || true
