---
layout: default
title: Getting Started
nav_order: 1
---

# Getting Started

Get up and running with SGEMM Optimization in under 5 minutes.

## Prerequisites

### Hardware

- **NVIDIA GPU**: Volta (sm_70) or newer architecture
- **RAM**: 4GB+ GPU memory recommended
- **Tested GPUs**:
  - RTX 3060 Laptop (Ampere, sm_86)
  - RTX 3090 (Ampere, sm_86)
  - A100 (Ampere, sm_80)
  - RTX 4090 (Ada, sm_89)

### Software

| Tool | Version | Purpose |
|------|---------|---------|
| CUDA Toolkit | 11.0+ | CUDA compiler and runtime |
| CMake | 3.18+ | Build system (recommended) |
| GCC/Clang | 9+ | Host compiler |
| Make | 4.0+ | Alternative build system |
| Google Test | 1.10+ | Unit testing (optional) |

## Quick Start

### Step 1: Clone the Repository

```bash
git clone https://github.com/LessUp/sgemm-optimization.git
cd sgemm-optimization
```

### Step 2: Build

**Option A: CMake (Recommended)**

```bash
# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)
```

**Option B: Makefile (Quick)**

```bash
# Build with specific GPU architecture
make GPU_ARCH=sm_86
```

### Step 3: Run Benchmark

```bash
# Default benchmark (aligned squares)
./build/bin/sgemm_benchmark

# All dimensions (including edge cases)
./build/bin/sgemm_benchmark -a

# Specific dimensions
./build/bin/sgemm_benchmark --dims 256 384 640

# Custom warmup/benchmark iterations
./build/bin/sgemm_benchmark -a --warmup 10 --benchmark 50
```

### Step 4: Run Tests

```bash
# Using CMake
cmake --build build --target test_sgemm
ctest --test-dir build

# Using Makefile
make test
```

## Expected Output

### Benchmark Output

```
================================================================================
Dimension: 1024x1024x1024
================================================================================
Kernel                  | GFLOPS   | vs cuBLAS | Time (ms) | Status
------------------------|----------|-----------|-----------|-------
cuBLAS                  |  5727    | 100.0%    |   0.375   | ✓
Tensor Core (WMMA)      |  2300    |  40.2%    |   0.934   | ✓
Tiled                   |   753    |  13.1%    |   2.853   | ✓
Double Buffer           |   701    |  12.2%    |   3.064   | ✓
Bank-Free               |   673    |  11.8%    |   3.190   | ✓
Naive                   |   604    |  10.6%    |   3.553   | ✓
================================================================================
```

### Test Output

```
[==========] Running 8 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 8 tests from SGEMMTest
[ RUN      ] SGEMMTest.NumericalCorrectness
[       OK ] SGEMMTest.NumericalCorrectness (45 ms)
[ RUN      ] SGEMMTest.TensorCoreFastPath
[       OK ] SGEMMTest.TensorCoreFastPath (12 ms)
...
[==========] 8 tests from 1 test suite ran. (156 ms total)
[  PASSED  ] 8 tests.
```

## Build Configuration

### GPU Architecture Flags

```bash
# Tesla V100 (Volta)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGPU_ARCH=sm_70

# RTX 3090 / A100 (Ampere)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGPU_ARCH=sm_86

# RTX 4090 (Ada Lovelace)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGPU_ARCH=sm_89

# H100 (Hopper)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGPU_ARCH=sm_90
```

### Build Types

| Type | Flags | Use Case |
|------|-------|----------|
| `Release` | `-O3 -DNDEBUG` | Benchmarking, production |
| `Debug` | `-g -G -DDEBUG` | Debugging with cuda-gdb |
| `RelWithDebInfo` | `-O2 -g -DNDEBUG` | Profiling with debug info |

## Troubleshooting

### Issue: "No CUDA runtime found"

**Solution**: Ensure `LD_LIBRARY_PATH` includes CUDA libraries:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Issue: "cuBLAS not found"

**Solution**: cuBLAS is included with CUDA Toolkit. Verify installation:

```bash
nvcc --version
ls /usr/local/cuda/lib64/libcublas*
```

### Issue: Build fails with "unsupported gpu architecture"

**Solution**: Use the correct `GPU_ARCH` for your GPU:

```bash
# Check your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Then build with matching architecture
make GPU_ARCH=sm_XX
```

### Issue: Tests fail with numerical errors

**Solution**: This is expected for Tensor Core kernels due to FP16 precision. Check tolerances:

- Standard kernels: `rtol=1e-3, atol=1e-4`
- Tensor Core: `rtol=5e-2, atol=1e-2`

## Next Steps

1. **Read the [Architecture Guide](architecture.md)** to understand system design
2. **Explore [Kernel Details](kernel-details.md)** for implementation deep-dives
3. **Review [Benchmark Results](benchmark-results.md)** for performance analysis
4. **Check [Specifications](../specs/)** for requirements and RFCs

## Additional Resources

- [GitHub Repository](https://github.com/LessUp/sgemm-optimization)
- [CHANGELOG](../CHANGELOG.md)
- [Contributing Guide](../CONTRIBUTING.md)
