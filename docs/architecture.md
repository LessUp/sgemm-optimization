---
layout: default
title: Architecture Guide
nav_order: 2
---

# Architecture Guide

This document describes the system architecture and design decisions for the SGEMM Optimization project.

## System Overview

The project implements five progressively optimized CUDA SGEMM (Single-precision General Matrix Multiply) kernels, demonstrating core GPU optimization techniques from a naive triple-loop to Tensor Core WMMA API usage.

### Design Principles

1. **Single Source of Truth**: All implementations follow specifications in `/specs/`
2. **Spec-Driven Development (SDD)**: Specs before code, documentation synchronized with implementation
3. **RAII Resource Management**: No raw `cudaFree()` calls, all resources use wrapper classes
4. **Exception-Based Error Handling**: No `exit()` in library code
5. **Header-Only Kernels**: All kernels are `.cuh` files for easy integration

## Architecture Layers

```
┌─────────────────────────────────────────────┐
│           Application Layer                  │
│  main.cu - Entry point & benchmark runner   │
├─────────────────────────────────────────────┤
│           Framework Layer                    │
│  utils/benchmark.cuh - Benchmark framework   │
│  utils/verify.cuh   - Correctness checks     │
│  utils/cuda_utils.cuh - RAII wrappers        │
├─────────────────────────────────────────────┤
│           Kernel Layer                       │
│  kernels/naive_sgemm.cuh                     │
│  kernels/tiled_sgemm.cuh                     │
│  kernels/bank_conflict_free_sgemm.cuh        │
│  kernels/double_buffer_sgemm.cuh             │
│  kernels/tensor_core_sgemm.cuh               │
├─────────────────────────────────────────────┤
│           CUDA Runtime Layer                 │
│  cuBLAS, CUDA Runtime API, WMMA API         │
└─────────────────────────────────────────────┘
```

## Unified Kernel Interface

All kernels implement the same interface template:

```cpp
template<int TILE_SIZE = 32>
void sgemm_naive(const float* A, const float* B, float* C,
                 int M, int N, int K, cudaStream_t stream = 0);

template<int TILE_SIZE = 32>
void sgemm_tiled(const float* A, const float* B, float* C,
                 int M, int N, int K, cudaStream_t stream = 0);

// ... and so on for each variant
```

### Interface Contract

| Parameter | Description | Constraints |
|-----------|-------------|-------------|
| `A` | Input matrix A (M×K), row-major | Device pointer, valid |
| `B` | Input matrix B (K×N), row-major | Device pointer, valid |
| `C` | Output matrix C (M×N), row-major | Device pointer, allocated |
| `M` | Rows of A and C | M > 0 |
| `N` | Columns of B and C | N > 0 |
| `K` | Columns of A / Rows of B | K > 0 |
| `stream` | CUDA stream (optional) | Default: 0 |

### Grid-Block Configuration

```cpp
dim3 block(TILE_SIZE, TILE_SIZE);
dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
          (M + TILE_SIZE - 1) / TILE_SIZE);
```

## Tensor Core Architecture

The Tensor Core kernel uses NVIDIA's WMMA (Warp Matrix Multiply Accumulate) API:

### Architecture-Specific Guards

```cpp
#if __CUDA_ARCH__ >= 700
    // WMMA code for Volta and newer
    #include <mma.h>
#else
    // Fallback to FP32 tiled kernel
#endif
```

### Alignment Requirements

| Requirement | Detail |
|-------------|--------|
| M, K, N | Must be multiples of 16 for WMMA fast path |
| Fallback | Automatic FP32 kernel for non-aligned sizes |
| Precision | FP16 input, FP32 accumulation |
| Verification | Relaxed tolerances: `rtol=5e-2, atol=1e-2` |

## Error Handling Strategy

All errors are handled via exceptions, not `exit()`:

```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error: ") + \
                cudaGetErrorString(err)); \
        } \
    } while(0)
```

## Memory Management

RAII wrappers ensure no memory leaks:

```cpp
struct DeviceBuffer {
    float* ptr;
    DeviceBuffer(size_t bytes) {
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
    }
    ~DeviceBuffer() {
        CUDA_CHECK(cudaFree(ptr));
    }
    // Deleted copy/move constructors prevent misuse
};
```

## Build System

### CMake (Recommended)

```cmake
add_executable(sgemm_benchmark src/main.cu)
target_link_libraries(sgemm_benchmark PRIVATE cublas)
set_target_properties(sgemm_benchmark PROPERTIES
    CUDA_ARCHITECTURES ${GPU_ARCH})
```

### GPU Architecture Support

| GPU | Architecture | Flag |
|-----|-------------|------|
| V100 | Volta | `sm_70` |
| RTX 3090/A100 | Ampere | `sm_86` |
| RTX 4090 | Ada | `sm_89` |
| H100 | Hopper | `sm_90` |

## References

- [RFC 0001: Core Architecture](../specs/rfc/0001-core-architecture.md)
- [RFC 0002: Implementation Roadmap](../specs/rfc/0002-implementation-roadmap.md)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
