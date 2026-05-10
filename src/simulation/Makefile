# ====== Build / Link paths ======
SUNDIALS_DIR := /usr/local/sundials
INCLUDE_FLAGS := -I$(SUNDIALS_DIR)/include
LIBRARY_FLAGS := -L$(SUNDIALS_DIR)/lib64

LDLIBS := -lsundials_cvode -lsundials_nveccuda \
          -lsundials_core -lcudart -lm

NVCC := nvcc
CUDA_ARCH ?= -gencode arch=compute_89,code=sm_89

# ====== Build mode ======
BUILD ?= release

# ====== Physics / grid knobs ======
PPW              ?= 20
DOMAIN_X_LAMBDA  ?= 6.0
DOMAIN_Y_LAMBDA  ?= 4.0
SCREEN_X_LAMBDA  ?= 1.5
SLIT_W_LAMBDA    ?= 1.0
SRC_X_LAMBDA     ?= 0.5
N_PERIODS        ?= 8.0
RAMP_PERIODS     ?= 1.5
CFL              ?= 0.5
OUTPUT_FRAMES    ?= 80

# ====== Solver knobs ======
RTOL_VAL   ?= 1.0e-5
ATOL_VAL   ?= 1.0e-7
KRYLOV_DIM ?= 5
BLOCK_SIZE ?= 256

ENABLE_OUTPUT ?= 1

ifeq ($(BUILD),debug)
  NVCC_FLAGS := $(CUDA_ARCH) -G -g -lineinfo -Xcompiler -O0
else
  NVCC_FLAGS := $(CUDA_ARCH) -O3 -lineinfo --use_fast_math
endif

NVCC_FLAGS += -DENABLE_OUTPUT=$(ENABLE_OUTPUT) \
              -DBLOCK_SIZE=$(BLOCK_SIZE) \
              -DRTOL_VAL=$(RTOL_VAL) -DATOL_VAL=$(ATOL_VAL) \
              -DKRYLOV_DIM=$(KRYLOV_DIM) \
              -DPPW=$(PPW) \
              -DDOMAIN_X_LAMBDA=$(DOMAIN_X_LAMBDA) \
              -DDOMAIN_Y_LAMBDA=$(DOMAIN_Y_LAMBDA) \
              -DSCREEN_X_LAMBDA=$(SCREEN_X_LAMBDA) \
              -DSLIT_W_LAMBDA=$(SLIT_W_LAMBDA) \
              -DSRC_X_LAMBDA=$(SRC_X_LAMBDA) \
              -DN_PERIODS=$(N_PERIODS) \
              -DRAMP_PERIODS=$(RAMP_PERIODS) \
              -DCFL=$(CFL) \
              -DOUTPUT_FRAMES=$(OUTPUT_FRAMES)

TARGET      := slit_fdtd
SRC         := slit_fdtd.cu
PRECOND_SRC := precond.cu
JTV_SRC     := jtv.cu

PRECOND_OBJ := precond.o
JTV_OBJ     := jtv.o

OBJS := $(PRECOND_OBJ) $(JTV_OBJ)

.PHONY: all run run-debug clean reallyclean show-config

all: $(TARGET)

$(PRECOND_OBJ): $(PRECOND_SRC) precond.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math \
	    $(INCLUDE_FLAGS) -c $< -o $@

$(JTV_OBJ): $(JTV_SRC) jtv.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math \
	    $(INCLUDE_FLAGS) -c $< -o $@

$(TARGET): $(SRC) $(OBJS) precond.h jtv.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -o $@ $< $(OBJS) \
	    $(LIBRARY_FLAGS) $(LDLIBS)

run: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET)

run-debug: $(TARGET)
	CUDA_LAUNCH_BLOCKING=1 LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET)

show-config:
	@echo "TARGET            = $(TARGET)"
	@echo "PPW               = $(PPW)"
	@echo "DOMAIN_X_LAMBDA   = $(DOMAIN_X_LAMBDA)"
	@echo "DOMAIN_Y_LAMBDA   = $(DOMAIN_Y_LAMBDA)"
	@echo "SCREEN_X_LAMBDA   = $(SCREEN_X_LAMBDA)"
	@echo "SLIT_W_LAMBDA     = $(SLIT_W_LAMBDA)"
	@echo "SRC_X_LAMBDA      = $(SRC_X_LAMBDA)"
	@echo "N_PERIODS         = $(N_PERIODS)"
	@echo "RAMP_PERIODS      = $(RAMP_PERIODS)"
	@echo "CFL               = $(CFL)"
	@echo "OUTPUT_FRAMES     = $(OUTPUT_FRAMES)"
	@echo "RTOL_VAL          = $(RTOL_VAL)"
	@echo "ATOL_VAL          = $(ATOL_VAL)"
	@echo "KRYLOV_DIM        = $(KRYLOV_DIM)"
	@echo "BLOCK_SIZE        = $(BLOCK_SIZE)"
	@echo "CUDA_ARCH         = $(CUDA_ARCH)"

clean:
	rm -f $(TARGET) $(OBJS)

reallyclean: clean
	rm -f output.bin slit_animation.mp4
