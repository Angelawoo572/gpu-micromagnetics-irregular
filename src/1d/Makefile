# ====== Build / Link paths ======
SUNDIALS_DIR := /usr/local/sundials
INCLUDE_FLAGS := -I$(SUNDIALS_DIR)/include
LIBRARY_FLAGS := -L$(SUNDIALS_DIR)/lib64

# cuFFTDx / MathDX
MATHDX_BASE := /opt/nvidia/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06
INCLUDE_FLAGS += -I$(MATHDX_BASE)/include -I$(MATHDX_BASE)/external/cutlass/include

LDLIBS := -lsundials_cvode -lsundials_nveccuda \
          -lsundials_sunlinsolcusolversp -lsundials_sunmatrixcusparse \
          -lsundials_core -lcusolver -lcusparse -lcudart -lcudadevrt -lm \
          -lcufft

NVCC := nvcc
CUDA_ARCH := -arch=sm_89
CUDA_DEBUG ?= -G   # 关掉调试: make CUDA_DEBUG=

TARGET := 1d
SRC := 1d.cu

ARGS ?=

NSYS ?= /opt/nvidia/nsight-systems/2025.1.3/bin/nsys
NCU  ?= /opt/nvidia/nsight-compute/2025.3.1/ncu

NSYS_UI ?= /usr/local/cuda/bin/nsight-sys

PROFILE_OUT ?= profile_$(TARGET)

NSYS_FLAGS_LIGHT  := profile --trace=cuda -o $(PROFILE_OUT)

NSYS_FLAGS_STD    := profile --trace=cuda --cuda-memory-usage=true -o $(PROFILE_OUT)


NSYS_FLAGS_EXT    := profile --trace=cuda --cuda-memory-usage=true \
                      --cuda-um-cpu-page-faults=true -o $(PROFILE_OUT)

NCU_FLAGS ?= --set full -o ncu_$(TARGET)

# ====== Phonies ======
.PHONY: all run profile profile-light profile-ext stats export-csv export-sqlite ui compute clean reallyclean

# ====== Build ======
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CUDA_ARCH) $(CUDA_DEBUG) $(INCLUDE_FLAGS) -o $@ $< $(LIBRARY_FLAGS) $(LDLIBS)

# ====== Run（======
run: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 ./$(TARGET) $(ARGS)

# ====== Nsight Systems ======
profile: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 \
	$(NSYS) $(NSYS_FLAGS_STD) ./$(TARGET) $(ARGS)
	@echo "Generated: $(PROFILE_OUT).nsys-rep"
	@echo "View stats: nsys stats --report system,cuda-apisum,gpu-kernsum,gpu-kernexec $(PROFILE_OUT).nsys-rep"

# ====== Nsight Systems ======
profile-light: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 \
	$(NSYS) $(NSYS_FLAGS_LIGHT) ./$(TARGET) $(ARGS)
	@echo "Generated: $(PROFILE_OUT).nsys-rep"

# ====== Nsight Systems ======
profile-ext: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 \
	$(NSYS) $(NSYS_FLAGS_EXT) ./$(TARGET) $(ARGS)
	@echo "Generated: $(PROFILE_OUT).nsys-rep"

# ====== Stats ======
stats:
	@[ -f "$(PROFILE_OUT).nsys-rep" ] || (echo "ERROR: $(PROFILE_OUT).nsys-rep not found. Run 'make profile' first."; exit 1)
	nsys stats --report system,cuda-apisum,gpu-kernsum,gpu-kernexec $(PROFILE_OUT).nsys-rep > nsys_report.txt
	@echo "Wrote: nsys_report.txt"

# ====== Export（CSV / SQLite）======
export-csv:
	@[ -f "$(PROFILE_OUT).nsys-rep" ] || (echo "ERROR: $(PROFILE_OUT).nsys-rep not found. Run 'make profile' first."; exit 1)
	nsys export --type csv --output ./csv_out $(PROFILE_OUT).nsys-rep
	@echo "CSV out -> ./csv_out/"

export-sqlite:
	@[ -f "$(PROFILE_OUT).nsys-rep" ] || (echo "ERROR: $(PROFILE_OUT).nsys-rep not found. Run 'make profile' first."; exit 1)
	nsys export --sqlite $(PROFILE_OUT).sqlite $(PROFILE_OUT).nsys-rep
	@echo "SQLite -> $(PROFILE_OUT).sqlite"

# ====== GUI（======
ui:
	@[ -f "$(PROFILE_OUT).nsys-rep" ] || (echo "ERROR: $(PROFILE_OUT).nsys-rep not found. Run 'make profile' first."; exit 1)
	$(NSYS_UI) $(PROFILE_OUT).nsys-rep

# ====== Nsight Compute======
compute: $(TARGET)
	LD_LIBRARY_PATH=$(SUNDIALS_DIR)/lib64 \
	$(NCU) $(NCU_FLAGS) ./$(TARGET) $(ARGS)
	@echo "Generated: ncu_$(TARGET).ncu-rep (open with: ncu-ui ncu_$(TARGET).ncu-rep)"

clean:
	rm -f $(TARGET)

reallyclean: clean
	rm -f $(PROFILE_OUT).nsys-rep $(PROFILE_OUT).qdrep $(PROFILE_OUT).sqlite ncu_$(TARGET).ncu-rep
	rm -rf csv_out
