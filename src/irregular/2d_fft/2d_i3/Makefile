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

BUILD ?= release
PRINT ?= 0

BLOCK_X ?= 16
BLOCK_Y ?= 8
RTOL_VAL ?= 1.0e-4
ATOL_VAL ?= 1.0e-4
T_TOTAL  ?= 1000.0
KRYLOV_DIM ?= 5

# Demag
DEMAG_STRENGTH ?= 1.0
DEMAG_THICK    ?= 1.0

# Grid
GRID_NX ?= 1536
GRID_NY ?= 512

# Antidot mesh geometry
N_HOLES_X ?= 4
N_HOLES_Y ?= 4
HOLE_SEED ?= 20251101
MASK_EPS_CELLS ?= 2.2

# Output schedule — set EARLY_SAVE_UNTIL=80 if you want fine early frames
EARLY_SAVE_UNTIL ?= 80
EARLY_SAVE_EVERY ?= 10
LATE_SAVE_EVERY  ?= 800

NUM_GRAINS      ?= 72
DEAD_GRAIN_FRAC ?= 0.16
GRAIN_Z_BIAS    ?= 1.6
IC_CORE_MZ      ?= 0.95

ifeq ($(BUILD),debug)
  NVCC_FLAGS := $(CUDA_ARCH) -G -g -lineinfo -Xcompiler -O0
else
  NVCC_FLAGS := $(CUDA_ARCH) -O3 -lineinfo --use_fast_math
endif

NVCC_FLAGS += -DENABLE_OUTPUT=$(PRINT) \
			  -DNUM_GRAINS=$(NUM_GRAINS) \
              -DDEAD_GRAIN_FRAC=$(DEAD_GRAIN_FRAC) \
              -DGRAIN_Z_BIAS=$(GRAIN_Z_BIAS) \
              -DIC_CORE_MZ=$(IC_CORE_MZ) \
              -DBLOCK_X=$(BLOCK_X) -DBLOCK_Y=$(BLOCK_Y) \
              -DRTOL_VAL=$(RTOL_VAL) -DATOL_VAL=$(ATOL_VAL) \
              -DT_TOTAL=$(T_TOTAL) \
              -DKRYLOV_DIM=$(KRYLOV_DIM) \
              -DEARLY_SAVE_UNTIL=$(EARLY_SAVE_UNTIL) \
              -DEARLY_SAVE_EVERY=$(EARLY_SAVE_EVERY) \
              -DLATE_SAVE_EVERY=$(LATE_SAVE_EVERY) \
              -DDEMAG_STRENGTH=$(DEMAG_STRENGTH) \
              -DDEMAG_THICK=$(DEMAG_THICK) \
              -DGRID_NX=$(GRID_NX) -DGRID_NY=$(GRID_NY) \
              -DN_HOLES_X=$(N_HOLES_X) -DN_HOLES_Y=$(N_HOLES_Y) \
              -DHOLE_SEED=$(HOLE_SEED) \
              -DMASK_EPS_CELLS=$(MASK_EPS_CELLS)

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
NSYS ?= nsys
NCU  ?= ncu

PROFILE_OUT ?= profile_$(TARGET)

NSYS_FLAGS := profile --trace=cuda --cuda-memory-usage=true -o $(PROFILE_OUT)
NCU_FLAGS  ?= --set roofline -o ncu_$(TARGET)

.PHONY: all run run-debug build-debug build-release \
        run-print run-noprint run-demag run-no-demag \
        profile stats compute show-config clean reallyclean

all: $(TARGET)

$(FUSED_OBJ): $(FUSED_SRC) deferred_nvector.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math $(INCLUDE_FLAGS) -c $< -o $@
$(PRECOND_OBJ): $(PRECOND_SRC) precond.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math $(INCLUDE_FLAGS) -c $< -o $@
$(JTV_OBJ): $(JTV_SRC) jtv.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math $(INCLUDE_FLAGS) -c $< -o $@
$(DEMAG_OBJ): $(DEMAG_SRC) demag_fft.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math $(INCLUDE_FLAGS) -c $< -o $@

$(TARGET): $(SRC) $(OBJS) deferred_nvector.h precond.h jtv.h demag_fft.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -o $@ $< $(OBJS) $(LIBRARY_FLAGS) $(LDLIBS)
build-release:
	$(MAKE) BUILD=release PRINT=$(PRINT)

build-debug:
	$(MAKE) BUILD=debug PRINT=$(PRINT)

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
	$(MAKE) DEMAG_STRENGTH=1.0 DEMAG_THICK=1.0 PRINT=$(PRINT)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS)

run-no-demag:
	$(MAKE) clean
	$(MAKE) DEMAG_STRENGTH=0.0 PRINT=$(PRINT)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS)

profile: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 \
	$(NSYS) $(NSYS_FLAGS) ./$(TARGET) $(ARGS)
	@echo "Generated: $(PROFILE_OUT).nsys-rep"

stats:
	@[ -f "$(PROFILE_OUT).nsys-rep" ] || (echo "ERROR: run make profile first."; exit 1)
	nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
	    $(PROFILE_OUT).nsys-rep > nsys_report.txt
	@echo "Wrote: nsys_report.txt"

compute: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 \
	$(NCU) $(NCU_FLAGS) ./$(TARGET) $(ARGS) || \
	(echo "Nsight Compute failed (likely permission issue)"; exit 1)

show-config:
	@echo "TARGET         = $(TARGET)"
	@echo "GRID           = $(GRID_NX) x $(GRID_NY)"
	@echo "HOLES          = $(N_HOLES_X) x $(N_HOLES_Y)  seed=$(HOLE_SEED), eps=$(MASK_EPS_CELLS)"
	@echo "DEMAG          = strength=$(DEMAG_STRENGTH) thick=$(DEMAG_THICK)"
	@echo "RTOL/ATOL      = $(RTOL_VAL)/$(ATOL_VAL)"
	@echo "T_TOTAL        = $(T_TOTAL)"
	@echo "KRYLOV_DIM     = $(KRYLOV_DIM)"
	@echo "PROFILE_OUT    = $(PROFILE_OUT)"

clean:
	rm -f $(TARGET) $(OBJS)

reallyclean: clean
	rm -f output.txt mask.txt
	rm -f $(PROFILE_OUT).nsys-rep $(PROFILE_OUT).sqlite
	rm -f ncu_$(TARGET).ncu-rep