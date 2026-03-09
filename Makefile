# SGEMM Optimization Makefile
# Supports CUDA compilation with cuBLAS linking

# CUDA compiler
NVCC = nvcc

# Detect GPU architecture (default to sm_86 for Ampere RTX 30 series)
GPU_ARCH ?= sm_86

# Compiler flags
NVCC_FLAGS = -O3 -std=c++17 -arch=$(GPU_ARCH) --use_fast_math
NVCC_FLAGS += -Xcompiler -Wall,-Wextra
NVCC_FLAGS += -I./src -I./src/kernels -I./src/utils

# Libraries
LIBS = -lcublas -lcurand

# Directories
SRC_DIR = src
KERNEL_DIR = $(SRC_DIR)/kernels
UTILS_DIR = $(SRC_DIR)/utils
TEST_DIR = tests
BUILD_DIR = build

# Source files
MAIN_SRC = $(SRC_DIR)/main.cu
TEST_SRC = $(TEST_DIR)/test_sgemm.cu

# Targets
MAIN_TARGET = $(BUILD_DIR)/sgemm_benchmark
TEST_TARGET = $(BUILD_DIR)/test_sgemm

# Header files (for dependency tracking)
HEADERS = $(wildcard $(KERNEL_DIR)/*.cuh) $(wildcard $(UTILS_DIR)/*.cuh)

.PHONY: all clean test dirs benchmark

all: dirs $(MAIN_TARGET)

dirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(KERNEL_DIR)
	@mkdir -p $(UTILS_DIR)
	@mkdir -p $(TEST_DIR)

$(MAIN_TARGET): $(MAIN_SRC) $(HEADERS) | dirs
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LIBS)

$(TEST_TARGET): $(TEST_SRC) $(HEADERS) | dirs
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LIBS) -lgtest -lgtest_main -lpthread

test: $(TEST_TARGET)
	./$(TEST_TARGET)

benchmark: $(MAIN_TARGET)
	./$(MAIN_TARGET)

clean:
	rm -rf $(BUILD_DIR)

# Debug build
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: all

# Profile build (for nsight)
profile: NVCC_FLAGS += -lineinfo
profile: all

# Help
help:
	@echo "SGEMM Optimization Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build main benchmark (default)"
	@echo "  test      - Build and run tests"
	@echo "  benchmark - Build and run benchmark"
	@echo "  clean     - Remove build artifacts"
	@echo "  debug     - Build with debug symbols"
	@echo "  profile   - Build with profiling info"
	@echo ""
	@echo "Variables:"
	@echo "  GPU_ARCH  - GPU architecture (default: sm_70)"
	@echo ""
	@echo "Example:"
	@echo "  make GPU_ARCH=sm_80 benchmark"
