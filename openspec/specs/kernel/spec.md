# Kernel Specification

> **Version**: 2.1.0 | **Last Updated**: 2026-04-16 | **Status**: Complete

## Purpose

Define the functional requirements for SGEMM kernel implementations covering five progressive optimization techniques from naive triple-loop to Tensor Core WMMA acceleration.

## Requirements

### Requirement: Kernel Implementations
The project SHALL implement five CUDA SGEMM kernel variants with progressive optimization techniques.

#### Scenario: Five kernel variants available
- **WHEN** a user builds the project
- **THEN** five kernel implementations SHALL be available: Naive, Tiled, Bank-Free, Double-Buffer, and Tensor Core

### Requirement: Correctness Verification
All kernels SHALL match cuBLAS reference output within specified tolerances.

#### Scenario: Standard FP32 kernels correctness
- **WHEN** any standard FP32 kernel (Naive, Tiled, Bank-Free, Double-Buffer) is executed
- **THEN** the output SHALL match cuBLAS with rtol=1e-3, atol=1e-4

#### Scenario: Tensor Core kernel correctness
- **WHEN** the Tensor Core kernel is executed with aligned dimensions
- **THEN** the output SHALL match cuBLAS with rtol=5e-2, atol=1e-2

---

## Detailed Requirements

### REQ-KERNEL-001: Kernel Implementations
**Status**: Active
**Priority**: High
**Source**: FR-1 (Product Requirements)

The project SHALL implement five CUDA SGEMM kernel variants:

| ID | Requirement | Status |
|----|-------------|--------|
| REQ-KERNEL-001.1 | **Naive Kernel**: Basic triple-loop, one output per thread | Complete |
| REQ-KERNEL-001.2 | **Tiled Kernel**: Shared memory blocking for data reuse | Complete |
| REQ-KERNEL-001.3 | **Bank-Free Kernel**: Padding to eliminate bank conflicts | Complete |
| REQ-KERNEL-001.4 | **Double-Buffer Kernel**: Dual buffers for compute/memory overlap | Complete |
| REQ-KERNEL-001.5 | **Tensor Core Kernel**: WMMA API for FP16→FP32 | Complete |

**Acceptance Criteria**:
- [x] All 5 kernel implementations complete and functional
- [x] Each kernel in separate file under `src/kernels/`

---

### REQ-KERNEL-002: Correctness Verification
**Status**: Active
**Priority**: High
**Source**: FR-2 (Product Requirements)

All kernels must match cuBLAS reference output.

**Tolerances**:

| Kernel Type | Relative Tolerance (rtol) | Absolute Tolerance (atol) |
|-------------|--------------------------|--------------------------|
| Standard FP32 (Naive, Tiled, Bank-Free, Double-Buffer) | 1e-3 | 1e-4 |
| Tensor Core (FP16→FP32 mixed precision) | 5e-2 | 1e-2 |

**Acceptance Criteria**:
- [x] All kernels pass cuBLAS comparison
- [x] Property-based tests cover 100+ dimension combinations

---

### REQ-KERNEL-003: Performance Benchmark
**Status**: Active
**Priority**: Medium
**Source**: FR-3 (Product Requirements)

The project shall provide benchmarking infrastructure to measure and compare kernel performance.

**Acceptance Criteria**:
- [x] CUDA Events-based timing for accurate GPU measurement
- [x] GFLOPS calculation and reporting
- [x] Performance comparison against cuBLAS baseline
- [x] Roofline model data export for performance analysis

---

### REQ-KERNEL-004: Build System Support
**Status**: Active
**Priority**: High
**Source**: FR-4 (Product Requirements)

**Acceptance Criteria**:
- [x] CMake build system (primary, recommended)
- [x] Makefile for quick local builds
- [x] Multi-GPU architecture support: sm_70, sm_75, sm_80, sm_86, sm_89, sm_90

---

## Constraints

### CON-KERNEL-001: CUDA Compatibility
- Target: CUDA 11.0+
- Compute Capability: 7.0+ (Volta through Hopper)
- Memory: Must operate within GPU memory limits

### CON-KERNEL-002: Build Systems
```bash
# CMake (primary)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Makefile (quick)
make GPU_ARCH=sm_86
```

### CON-KERNEL-003: Educational Focus
- Code MUST be readable and well-commented
- Each optimization level MUST be independently compilable
- Progressive complexity for learning purposes

---

## Benchmark Configuration

Default benchmark matrix includes:

- **Aligned square cases**: `512×512×512`, `1024×1024×1024`
- **Aligned non-square case**: `256×384×640`
- **Unaligned edge case**: `511×513×1025` (exercises Tensor Core fallback path)

---

## Performance Expectations

| Stage | Kernel | Technique | Expected Speedup |
|-------|--------|-----------|------------------|
| 1 | Naive | Baseline | 1× |
| 2 | Tiled | Shared memory blocking | ~1.2-1.5× |
| 3 | Bank-Free | Bank conflict elimination | ~1.1× over Tiled |
| 4 | Double-Buffer | Compute/memory overlap | ~1.1× over Bank-Free |
| 5 | Tensor Core | WMMA API | ~3-4× over Double-Buffer |

---

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [WMMA API Reference](https://docs.nvidia.com/cuda/cuda-c-api/group__CUDA__WMMA.html)
- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA's high-performance GEMM library
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) — Simon Boehm
