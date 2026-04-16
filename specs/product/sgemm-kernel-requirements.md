# SGEMM Kernel Product Requirements

> **Version**: 2.1.0 | **Last Updated**: 2026-04-16 | **Status**: Complete

## Overview

This document defines the product requirements for the SGEMM (Single-precision General Matrix Multiplication) Optimization project. The project demonstrates progressive GPU kernel optimization techniques through five implementation stages, from a naive triple-loop to Tensor Core WMMA acceleration.

## Functional Requirements

### FR-1: Kernel Implementations

The project shall implement five CUDA SGEMM kernel variants, each building upon the previous optimization technique.

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-1.1 | **Naive Kernel**: Each thread computes one output element using a basic triple-loop approach | High | ✅ Complete |
| FR-1.2 | **Tiled Kernel**: Implement shared memory blocking to improve data reuse and reduce global memory traffic | High | ✅ Complete |
| FR-1.3 | **Bank-Free Kernel**: Apply shared memory padding (e.g., `[32][33]`) to eliminate bank conflicts | High | ✅ Complete |
| FR-1.4 | **Double-Buffer Kernel**: Implement dual shared-memory buffers to overlap computation with memory transfers | High | ✅ Complete |
| FR-1.5 | **Tensor Core Kernel**: Use WMMA API for mixed-precision FP16→FP32 matrix multiply-accumulate | High | ✅ Complete |

### FR-2: Correctness Verification

All kernel implementations must produce results consistent with NVIDIA's cuBLAS library.

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-2.1 | All kernels must match cuBLAS reference output | High | ✅ Complete |
| FR-2.2 | Standard kernels (FP32): tolerance `rtol=1e-3, atol=1e-4` | High | ✅ Complete |
| FR-2.3 | Tensor Core kernel (FP16→FP32): tolerance `rtol=5e-2, atol=1e-2` | High | ✅ Complete |

### FR-3: Performance Benchmark

The project shall provide benchmarking infrastructure to measure and compare kernel performance.

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-3.1 | CUDA Events-based timing for accurate GPU measurement | High | ✅ Complete |
| FR-3.2 | GFLOPS calculation and reporting | High | ✅ Complete |
| FR-3.3 | Performance comparison against cuBLAS baseline | Medium | ✅ Complete |
| FR-3.4 | Roofline model data export for performance analysis | Low | ✅ Complete |

### FR-4: Build System

The project shall support multiple build systems and GPU architectures.

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-4.1 | CMake build system (primary, recommended) | High | ✅ Complete |
| FR-4.2 | Makefile for quick local builds | Medium | ✅ Complete |
| FR-4.3 | Multi-GPU architecture support: `sm_70`, `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90` | High | ✅ Complete |

## Non-Functional Requirements

### NFR-1: Code Quality

| ID | Requirement | Status |
|----|-------------|--------|
| NFR-1.1 | RAII memory management throughout; zero memory leaks | ✅ Complete |
| NFR-1.2 | Exception-based error handling; no `exit()` calls in library code | ✅ Complete |
| NFR-1.3 | clang-format code style enforced via CI | ✅ Complete |

### NFR-2: Testing

| ID | Requirement | Status |
|----|-------------|--------|
| NFR-2.1 | Google Test unit tests for all kernels | ✅ Complete |
| NFR-2.2 | Property-based tests covering square, non-square, and edge-case dimensions | ✅ Complete |
| NFR-2.3 | GitHub Actions CI workflow for automated checks | ✅ Complete |

### NFR-3: Documentation

| ID | Requirement | Status |
|----|-------------|--------|
| NFR-3.1 | Bilingual README (English primary, Chinese secondary) | ✅ Complete |
| NFR-3.2 | CHANGELOG following Keep a Changelog format | ✅ Complete |
| NFR-3.3 | Code comments explaining optimization principles | ✅ Complete |

## Acceptance Criteria

All criteria have been met as of version 2.1.0:

1. ✅ All 5 kernel implementations complete and functional
2. ✅ Correctness verification passes against cuBLAS for all kernels
3. ✅ CMake and Make builds succeed on supported GPU architectures
4. ✅ Google Test suite passes (unit tests + property tests)
5. ✅ CI workflow runs successfully on push/PR

## Benchmark Configuration

Default benchmark matrix includes:

- **Aligned square cases**: `512×512×512`, `1024×1024×1024`
- **Aligned non-square case**: `256×384×640`
- **Unaligned edge case**: `511×513×1025` (exercises Tensor Core fallback path)

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) — Simon Boehm
- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA's high-performance GEMM library
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
