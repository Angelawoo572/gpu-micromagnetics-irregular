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

NX_VAL ?= 1536
NY_VAL ?= 512

RTOL_VAL ?= 1.0e-4
ATOL_VAL ?= 1.0e-4
T_TOTAL  ?= 1000.0

# KRYLOV_DIM: max Krylov subspace dimension for SPGMR.
#   0 = SUNDIALS default (min(neq, 5)).
#  15 = more GMRES headroom before restart.  Profile at dim=5 showed
#       nli=75,314 with average 3.2 iters/solve but ~30% of solves hit
#       the restart limit.  dim=15 lets those continue instead of restarting.
#       Cost: 15 N_Vectors (~22 MB) vs 5 (~7.5 MB).  Easily fits.
KRYLOV_DIM ?= 5

# ====== FFT Demagnetization (professor's Newell calt/ctt) ======
# DEMAG_STRENGTH = 0.0  → disabled (exchange-only, original behaviour)
# DEMAG_STRENGTH = 1.0  → full dipolar demag via cuFFT Z2Z
# DEMAG_THICK           → cell thickness in z (in cell-spacing units)
DEMAG_STRENGTH ?= 2.0
DEMAG_THICK    ?= 1.0

# ====== Output schedule ======
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
			  -DNX_VAL=$(NX_VAL) \
              -DNY_VAL=$(NY_VAL) \
              -DKRYLOV_DIM=$(KRYLOV_DIM) \
              -DEARLY_SAVE_UNTIL=$(EARLY_SAVE_UNTIL) \
              -DEARLY_SAVE_EVERY=$(EARLY_SAVE_EVERY) \
              -DLATE_SAVE_EVERY=$(LATE_SAVE_EVERY) \
              -DDEMAG_STRENGTH=$(DEMAG_STRENGTH) \
              -DDEMAG_THICK=$(DEMAG_THICK) \

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
        sweep-block sweep-tol sweep-demag sweep-krylov \
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

# Professor's Newell tensor (calt/ctt) + cuFFT Z2Z convolution (batched)
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
	$(MAKE) DEMAG_STRENGTH=2.0 DEMAG_THICK=1.0
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

sweep-demag:
	@for s in 0.0 0.01 0.1 0.5 1.0; do \
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
	@echo "KRYLOV_DIM     = $(KRYLOV_DIM)  (0=SUNDIALS default=5)"
	@echo "RTOL_VAL       = $(RTOL_VAL)"
	@echo "ATOL_VAL       = $(ATOL_VAL)"
	@echo "T_TOTAL        = $(T_TOTAL)"
	@echo "BLOCK_X        = $(BLOCK_X)   BLOCK_Y = $(BLOCK_Y)"
	@echo "CUDA_ARCH      = $(CUDA_ARCH)"
	@echo "PROFILE_OUT    = $(PROFILE_OUT)"


# ====== Direct (O(N²)) target — drop-in swap of demag module ======
DIRECT_TARGET  := 2d_direct
DIRECT_DEMAG_SRC := demag_direct.cu
DIRECT_DEMAG_OBJ := demag_direct.o
DIRECT_OBJS    := $(FUSED_OBJ) $(PRECOND_OBJ) $(JTV_OBJ) $(DIRECT_DEMAG_OBJ)

$(DIRECT_DEMAG_OBJ): $(DIRECT_DEMAG_SRC) demag_fft.h
	$(NVCC) $(CUDA_ARCH) -O3 --use_fast_math \
	    $(INCLUDE_FLAGS) -c $< -o $@

$(DIRECT_TARGET): 2d_fft.cu $(DIRECT_OBJS) deferred_nvector.h precond.h jtv.h demag_fft.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -o $@ 2d_fft.cu $(DIRECT_OBJS) \
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
	@echo "Direct version (O(N²)):"
	@echo "========================================================"
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(DIRECT_TARGET) $(ARGS)

.PHONY: run-direct bench sweep-direct-fft-size sweep-direct-fft-size-big

sweep-direct-fft-size-big:
	@mkdir -p scaling_logs
	@echo "size,nx,ny,ng,ncell,fft_grid,method,time_ms,nst,nfe,nli,ncfn,netf,speedup_vs_fft" > scaling_logs/summary.csv
	@for ny in 32 64 96 128 192 256 384 512 768 1024; do \
	  nx=`expr $$ny \* 3`; \
	  ng=$$ny; \
	  ncell=`expr $$ng \* $$ny`; \
	  echo "========================================================"; \
	  echo "Testing nx=$$nx ny=$$ny ng=$$ng ncell=$$ncell FFT=$${ny}x$${ny}"; \
	  echo "========================================================"; \
	  $(MAKE) clean >/dev/null 2>&1; \
	  $(MAKE) BUILD=release PRINT=0 NX_VAL=$$nx NY_VAL=$$ny DEMAG_STRENGTH=2.0 DEMAG_THICK=1.0 $(TARGET) >/dev/null; \
	  echo "[FFT] nx=$$nx ny=$$ny ng=$$ng ncell=$$ncell FFT=$${ny}x$${ny}" | tee scaling_logs/fft_$${nx}x$${ny}.txt; \
	  LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS) | tee -a scaling_logs/fft_$${nx}x$${ny}.txt; \
	  fft_time=`grep "GPU simulation took" scaling_logs/fft_$${nx}x$${ny}.txt | awk '{print $$4}'`; \
	  fft_stats1=`grep "nst =" scaling_logs/fft_$${nx}x$${ny}.txt | tail -1`; \
	  fft_stats2=`grep "nli =" scaling_logs/fft_$${nx}x$${ny}.txt | tail -1`; \
	  fft_nst=`echo $$fft_stats1 | awk '{print $$3}'`; \
	  fft_nfe=`echo $$fft_stats1 | awk '{print $$6}'`; \
	  fft_ncfn=`echo $$fft_stats1 | awk '{print $$15}'`; \
	  fft_netf=`echo $$fft_stats1 | awk '{print $$18}'`; \
	  fft_nli=`echo $$fft_stats2 | awk '{print $$3}'`; \
	  echo "$${nx}x$${ny},$$nx,$$ny,$$ng,$$ncell,$${ny}x$${ny},FFT,$$fft_time,$$fft_nst,$$fft_nfe,$$fft_nli,$$fft_ncfn,$$fft_netf,1.000" >> scaling_logs/summary.csv; \
	  echo ""; \
	  $(MAKE) $(DIRECT_TARGET) BUILD=release PRINT=0 NX_VAL=$$nx NY_VAL=$$ny DEMAG_STRENGTH=2.0 DEMAG_THICK=1.0 >/dev/null; \
	  echo "[DIRECT] nx=$$nx ny=$$ny ng=$$ng ncell=$$ncell FFT=$${ny}x$${ny}" | tee scaling_logs/direct_$${nx}x$${ny}.txt; \
	  LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(DIRECT_TARGET) $(ARGS) | tee -a scaling_logs/direct_$${nx}x$${ny}.txt; \
	  direct_time=`grep "GPU simulation took" scaling_logs/direct_$${nx}x$${ny}.txt | awk '{print $$4}'`; \
	  direct_stats1=`grep "nst =" scaling_logs/direct_$${nx}x$${ny}.txt | tail -1`; \
	  direct_stats2=`grep "nli =" scaling_logs/direct_$${nx}x$${ny}.txt | tail -1`; \
	  direct_nst=`echo $$direct_stats1 | awk '{print $$3}'`; \
	  direct_nfe=`echo $$direct_stats1 | awk '{print $$6}'`; \
	  direct_ncfn=`echo $$direct_stats1 | awk '{print $$15}'`; \
	  direct_netf=`echo $$direct_stats1 | awk '{print $$18}'`; \
	  direct_nli=`echo $$direct_stats2 | awk '{print $$3}'`; \
	  speedup=`awk -v d=$$direct_time -v f=$$fft_time 'BEGIN { if (f > 0) printf "%.3f", d/f; else print "nan" }'`; \
	  echo "$${nx}x$${ny},$$nx,$$ny,$$ng,$$ncell,$${ny}x$${ny},Direct,$$direct_time,$$direct_nst,$$direct_nfe,$$direct_nli,$$direct_ncfn,$$direct_netf,$$speedup" >> scaling_logs/summary.csv; \
	  echo "SPEEDUP nx=$$nx ny=$$ny FFT=$${ny}x$${ny}: $$speedup x" | tee scaling_logs/speedup_$${nx}x$${ny}.txt; \
	done
	@echo "Wrote scaling_logs/summary.csv"

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
