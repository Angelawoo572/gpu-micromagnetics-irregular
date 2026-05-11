# ===== Paths =====
SUNDIALS_DIR := /usr/local/sundials
INCLUDE_FLAGS := -I$(SUNDIALS_DIR)/include
LIBRARY_FLAGS := -L$(SUNDIALS_DIR)/lib64
LDLIBS := -lsundials_cvode -lsundials_nveccuda \
          -lsundials_core -lcudart -lcufft -lm

NVCC       := nvcc
CUDA_ARCH  ?= -gencode arch=compute_89,code=sm_89
BUILD      ?= release

# ===== Physics knobs =====
NX_VAL          ?= 768
NY_VAL          ?= 256
SCREEN_X_FRAC   ?= 0.50
SLIT_W_FRAC     ?= 0.10
SRC_X_FRAC      ?= 0.25
DRIVE_FREQ      ?= 0.12
H_DRIVE         ?= 0.10
RAMP_TIME       ?= 30.0
C_ALPHA         ?= 0.005
C_CHE           ?= 4.0
C_CHK           ?= 0.5
C_CHB           ?= 0.1
DEMAG_STRENGTH  ?= 1.0
DEMAG_THICK     ?= 1.0
PML_CELLS       ?= 20
PML_ALPHA_MAX   ?= 0.5

# ===== Solver knobs =====
RTOL_VAL        ?= 1.0e-4
ATOL_VAL        ?= 1.0e-6
KRYLOV_DIM      ?= 5
BLOCK_SIZE      ?= 256
BLOCK_X         ?= 16
BLOCK_Y         ?= 16
T_TOTAL         ?= 500.0
T1              ?= 5.0
ENABLE_OUTPUT   ?= 1
PRINT           ?= 0

ifeq ($(BUILD),debug)
  NVCC_FLAGS := $(CUDA_ARCH) -G -g -lineinfo -Xcompiler -O0
else
  NVCC_FLAGS := $(CUDA_ARCH) -O3 -lineinfo --use_fast_math
endif

NVCC_FLAGS += \
  -DNX_VAL=$(NX_VAL) -DNY_VAL=$(NY_VAL) \
  -DSCREEN_X_FRAC=$(SCREEN_X_FRAC) \
  -DSLIT_W_FRAC=$(SLIT_W_FRAC) \
  -DSRC_X_FRAC=$(SRC_X_FRAC) \
  -DDRIVE_FREQ=$(DRIVE_FREQ) \
  -DH_DRIVE=$(H_DRIVE) \
  -DRAMP_TIME=$(RAMP_TIME) \
  -DC_ALPHA=$(C_ALPHA) \
  -DC_CHE=$(C_CHE) \
  -DC_CHK=$(C_CHK) \
  -DC_CHB=$(C_CHB) \
  -DDEMAG_STRENGTH=$(DEMAG_STRENGTH) \
  -DDEMAG_THICK=$(DEMAG_THICK) \
  -DPML_CELLS=$(PML_CELLS) \
  -DPML_ALPHA_MAX=$(PML_ALPHA_MAX) \
  -DRTOL_VAL=$(RTOL_VAL) \
  -DATOL_VAL=$(ATOL_VAL) \
  -DKRYLOV_DIM=$(KRYLOV_DIM) \
  -DBLOCK_SIZE=$(BLOCK_SIZE) \
  -DBLOCK_X=$(BLOCK_X) \
  -DBLOCK_Y=$(BLOCK_Y) \
  -DT_TOTAL=$(T_TOTAL) \
  -DT1=$(T1) \
  -DENABLE_OUTPUT=$(ENABLE_OUTPUT) \
  -DPRINT=$(PRINT)

TARGET  := slit_spinwave
SRCS    := 2d_slit.cu precond.cu jtv.cu

# demag_fft.cu must be copied from i2 (shared implementation)
DEMAG_SRC := demag_fft.cu
DEMAG_OBJ := demag_fft.o
PRECOND_OBJ := precond.o
JTV_OBJ     := jtv.o

.PHONY: all run clean show-config

all: $(TARGET)

$(DEMAG_OBJ): $(DEMAG_SRC) demag_fft.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math $(INCLUDE_FLAGS) -c $< -o $@

$(PRECOND_OBJ): precond.cu precond.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math \
	    $(INCLUDE_FLAGS) $(NVCC_FLAGS) -c $< -o $@

$(JTV_OBJ): jtv.cu jtv.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math \
	    $(INCLUDE_FLAGS) $(NVCC_FLAGS) -c $< -o $@

$(TARGET): 2d_slit.cu $(PRECOND_OBJ) $(JTV_OBJ) $(DEMAG_OBJ) precond.h jtv.h demag_fft.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -o $@ 2d_slit.cu \
	    $(PRECOND_OBJ) $(JTV_OBJ) $(DEMAG_OBJ) \
	    $(LIBRARY_FLAGS) $(LDLIBS)

run: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET)

clean:
	rm -f $(TARGET) $(PRECOND_OBJ) $(JTV_OBJ) $(DEMAG_OBJ)

show-config:
	@echo "NX_VAL=$(NX_VAL)  NY_VAL=$(NY_VAL)"
	@echo "SCREEN_X_FRAC=$(SCREEN_X_FRAC)  SLIT_W_FRAC=$(SLIT_W_FRAC)"
	@echo "DRIVE_FREQ=$(DRIVE_FREQ)  H_DRIVE=$(H_DRIVE)"
	@echo "C_ALPHA=$(C_ALPHA)  C_CHE=$(C_CHE)  C_CHK=$(C_CHK)"
	@echo "DEMAG_STRENGTH=$(DEMAG_STRENGTH)  T_TOTAL=$(T_TOTAL)"
