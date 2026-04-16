# RFC 0001: Core Architecture

> **Status**: Accepted | **Created**: 2026-04-16 | **Last Updated**: 2026-04-16
> **Author**: Project Contributors

## Summary

This RFC describes the architectural design of the CUDA SGEMM Optimization system, which implements a progressive optimization path from a naive matrix multiplication kernel to a Tensor Core WMMA-accelerated implementation.

## Context

The project serves as an educational and demonstrative implementation of core GPU optimization techniques in the HPC domain. The architecture must support five distinct kernel implementations while maintaining a clean, testable, and benchmarkable codebase.

## Architecture Overview

The system follows a three-layer architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.cu                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Benchmark  │  │   Verify    │  │  CLI Parser │              │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘              │
└─────────┼────────────────┼──────────────────────────────────────┘
          │                │
          ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Kernel Implementations                        │
│  ┌────────┐ ┌────────┐ ┌────────────┐ ┌─────────────┐ ┌───────┐ │
│  │ Naive  │ │ Tiled  │ │ Bank-Free  │ │ Dbl-Buffer  │ │ TC    │ │
│  └────────┘ └────────┘ └────────────┘ └─────────────┘ └───────┘ │
└─────────────────────────────────────────────────────────────────┘
          │                │
          ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      cuBLAS Reference                            │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 1: Application Layer (`main.cu`)

The entry point orchestrates:
- **Benchmark Module**: CUDA Events-based timing for accurate GPU performance measurement
- **Verification Module**: Correctness checking against cuBLAS reference implementation
- **CLI Parser**: Command-line interface for dimension specification and kernel selection

### Layer 2: Kernel Layer (`src/kernels/`)

Five kernel implementations, each building upon previous optimization techniques:

| Kernel | File | Optimization Technique |
|--------|------|----------------------|
| Naive | `naive_sgemm.cuh` | Basic triple-loop; baseline implementation |
| Tiled | `tiled_sgemm.cuh` | Shared memory blocking for data reuse |
| Bank-Free | `bank_conflict_free_sgemm.cuh` | Shared memory padding to eliminate bank conflicts |
| Double-Buffer | `double_buffer_sgemm.cuh` | Dual buffers to overlap compute and memory transfers |
| Tensor Core | `tensor_core_sgemm.cuh` | WMMA API for mixed-precision FP16→FP32 compute |

### Layer 3: Utility Layer (`src/utils/`)

Supporting infrastructure:

| Module | File | Purpose |
|--------|------|---------|
| CUDA Utilities | `cuda_utils.cuh` | RAII wrappers, error handling macros, helper functions |
| Benchmark Framework | `benchmark.cuh` | Performance measurement using CUDA Events |
| Verification | `verify.cuh` | Numerical correctness checking vs cuBLAS |

## Kernel Interface Design

All kernels conform to a unified template interface:

```cpp
template<int TILE_SIZE = 32>
void launch_xxx_sgemm(
    const float* A,      // M×K input matrix
    const float* B,      // K×N input matrix
    float* C,            // M×N output matrix
    int M, int K, int N,
    cudaStream_t stream = 0
);
```

This consistency enables:
- Seamless kernel swapping in benchmark code
- Uniform error handling and resource management
- Simplified testing across all implementations

## Error Handling Strategy

The project uses exception-based error handling to ensure RAII cleanup correctness:

```cpp
struct CudaError : std::runtime_error {
    explicit CudaError(const std::string& msg) : std::runtime_error(msg) {}
};

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw CudaError(cudaGetErrorString(err)); \
        } \
    } while(0)
```

**Key Principle**: No `exit()` calls in library code; all errors propagate via exceptions to ensure destructors run.

## Testing Architecture

| Test Type | Framework | Coverage |
|-----------|-----------|----------|
| Unit Tests | Google Test | Boundary conditions, special values, error cases |
| Property Tests | GTest + Random | 100+ random dimension combinations |
| Verification | vs cuBLAS | `allclose` with kernel-specific tolerances |

### Verification Tolerances

| Kernel Type | Relative Tolerance (rtol) | Absolute Tolerance (atol) |
|-------------|--------------------------|--------------------------|
| Standard FP32 (Naive, Tiled, Bank-Free, Double-Buffer) | 1e-3 | 1e-4 |
| Tensor Core (FP16→FP32 mixed precision) | 5e-2 | 1e-2 |

## Build System Design

### Primary: CMake

```cmake
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Features:
- Multi-architecture support via `GPU_ARCH` CMake variable
- Google Test integration
- Clean separation of build targets

### Quick Start: Makefile

```bash
make GPU_ARCH=sm_86
make benchmark
make test
```

### Supported GPU Architectures

| GPU | Architecture | Compute Capability | Build Flag |
|-----|-------------|-------------------|------------|
| Tesla V100 | Volta | sm_70 | `GPU_ARCH=sm_70` |
| RTX 2080 | Turing | sm_75 | `GPU_ARCH=sm_75` |
| RTX 3090 / A100 | Ampere | sm_80 / sm_86 | `GPU_ARCH=sm_86` |
| RTX 4090 / L40 | Ada Lovelace | sm_89 | `GPU_ARCH=sm_89` |
| H100 | Hopper | sm_90 | `GPU_ARCH=sm_90` |

## Performance Expectations

| Stage | Kernel | Technique | Expected Speedup |
|-------|--------|-----------|------------------|
| 1 | Naive | Baseline | 1× |
| 2 | Tiled | Shared memory blocking | ~1.2-1.5× |
| 3 | Bank-Free | Bank conflict elimination | ~1.1× over Tiled |
| 4 | Double-Buffer | Compute/memory overlap | ~1.1× over Bank-Free |
| 5 | Tensor Core | WMMA API | ~3-4× over Double-Buffer |

## Project Structure

```
sgemm-optimization/
├── src/
│   ├── kernels/
│   │   ├── naive_sgemm.cuh              # Baseline implementation
│   │   ├── tiled_sgemm.cuh              # Shared memory blocking
│   │   ├── bank_conflict_free_sgemm.cuh # Bank conflict elimination
│   │   ├── double_buffer_sgemm.cuh      # Double buffer pipeline
│   │   └── tensor_core_sgemm.cuh        # Tensor Core WMMA
│   ├── utils/
│   │   ├── cuda_utils.cuh               # RAII, error handling
│   │   ├── benchmark.cuh                # Performance measurement
│   │   └── verify.cuh                   # Correctness verification
│   └── main.cu                          # Entry point
├── tests/
│   └── test_sgemm.cu                    # Google Test suite
├── specs/                               # Specification documents
│   ├── product/                         # Product requirements
│   ├── rfc/                             # Technical design documents
│   └── testing/                         # Test specifications
├── .github/workflows/
│   ├── ci.yml                           # CI: format + build checks
│   └── pages.yml                        # GitHub Pages deployment
├── CHANGELOG.md                         # Version history
├── CMakeLists.txt                       # CMake build (recommended)
├── Makefile                             # Make build (quick start)
├── README.md                            # English documentation
└── README.zh-CN.md                      # Chinese documentation
```

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [WMMA API Reference](https://docs.nvidia.com/cuda/cuda-c-api/group__CUDA__WMMA.html)
- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA's high-performance GEMM library
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) — Simon Boehm
