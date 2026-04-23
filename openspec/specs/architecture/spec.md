# Architecture Specification

> **Version**: 2.1.0 | **Last Updated**: 2026-04-23 | **Status**: Complete

## Purpose

Define system architecture decisions and implementation roadmap for the SGEMM optimization project. This document consolidates RFC 0001 (Core Architecture) and RFC 0002 (Implementation Roadmap).

## Requirements

### Requirement: Three-Layer Architecture
The system SHALL follow a three-layer architecture pattern for clean separation of concerns.

#### Scenario: Application layer provides user interface
- **WHEN** a user runs the benchmark or verification
- **THEN** the application layer SHALL orchestrate benchmark, verification, and CLI parsing

#### Scenario: Kernel layer provides implementations
- **WHEN** matrix multiplication is requested
- **THEN** the kernel layer SHALL provide five progressive optimization implementations

#### Scenario: Utility layer provides infrastructure
- **WHEN** any kernel executes
- **THEN** the utility layer SHALL provide RAII memory management and error handling

### Requirement: Unified Kernel Interface
All kernels SHALL conform to a unified template interface for seamless swapping.

#### Scenario: Consistent kernel launch signature
- **WHEN** a kernel is invoked
- **THEN** it SHALL accept (A, B, C, M, K, N, stream) parameters with consistent types

### Requirement: Published architecture matches the real repository structure
The repository architecture guidance MUST describe only the directory structure, documentation boundaries, and engineering surfaces that actually exist and are maintained.

#### Scenario: Contributor consults architecture guidance
- **WHEN** a contributor reads architecture-facing documentation or specifications
- **THEN** all referenced repository paths, layers, and responsibilities MUST correspond to the real maintained layout and MUST NOT reference stale or superseded structures as authoritative

### Requirement: Engineering boundaries are explicit
The repository architecture MUST make local-only and CI-safe responsibilities explicit so maintainers can reason correctly about build, test, and validation coverage.

#### Scenario: Contributor decides how to validate a change
- **WHEN** a contributor evaluates required validation steps for code, docs, specs, or workflow changes
- **THEN** the architecture guidance MUST clearly distinguish local GPU-dependent verification from CI-safe compile, structure, and publication checks

---

## Design Decisions
**Date**: 2026-04-16
**Status**: Active
**Source**: RFC 0001

**Decision**: System follows three layers:

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

1. **Application Layer** (`main.cu`) - Benchmark, Verify, CLI
2. **Kernel Layer** (`src/kernels/`) - 5 kernel implementations
3. **Utility Layer** (`src/utils/`) - RAII, error handling, verification

**Rationale**: Clean separation enables independent testing and benchmarking.

---

### DEC-ARCH-002: Unified Kernel Interface
**Date**: 2026-04-16
**Status**: Active
**Source**: RFC 0001

**Decision**: All kernels conform to a unified template interface:

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

**Rationale**: Enables seamless kernel swapping and uniform testing.

---

### DEC-ARCH-003: Exception-Based Error Handling
**Date**: 2026-04-16
**Status**: Active
**Source**: RFC 0001

**Decision**: Use exceptions, not `exit()`, for error handling.

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

**Rationale**: Ensures RAII cleanup correctness; destructors always run.

---

### DEC-ARCH-004: Kernel Organization
**Date**: 2026-04-16
**Status**: Active
**Source**: RFC 0001

**Decision**: Each optimization level has separate kernel file:

| Kernel | File | Optimization Technique |
|--------|------|----------------------|
| Naive | `naive_sgemm.cuh` | Basic triple-loop; baseline implementation |
| Tiled | `tiled_sgemm.cuh` | Shared memory blocking for data reuse |
| Bank-Free | `bank_conflict_free_sgemm.cuh` | Shared memory padding to eliminate bank conflicts |
| Double-Buffer | `double_buffer_sgemm.cuh` | Dual buffers to overlap compute and memory transfers |
| Tensor Core | `tensor_core_sgemm.cuh` | WMMA API for mixed-precision FP16→FP32 compute |

**Rationale**: Enables incremental learning and allows performance comparison between approaches.

---

## Implementation Roadmap

### Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project Infrastructure | Complete |
| 2 | Kernel Implementation | Complete |
| 3 | Utility Infrastructure | Complete |
| 4 | Testing Suite | Complete |
| 5 | Build System & CI/CD | Complete |
| 6 | Documentation | Complete |
| 7 | Code Quality & Refinement | Complete |

**Current Version**: 2.1.0 - All phases complete.

---

### Phase 1: Project Infrastructure ✅

**Completed Tasks**:
- [x] Project directory structure setup
- [x] `.gitignore` configuration (CUDA/Profiling/IDE rules)
- [x] `.editorconfig` for consistent code formatting
- [x] MIT LICENSE file

**Deliverables**: `.gitignore`, `.editorconfig`, `LICENSE`

---

### Phase 2: Kernel Implementation ✅

**Completed Tasks**:
- [x] `naive_sgemm.cuh` — Basic triple-loop baseline implementation
- [x] `tiled_sgemm.cuh` — Shared memory blocking for data reuse
- [x] `bank_conflict_free_sgemm.cuh` — Shared memory padding to eliminate bank conflicts
- [x] `double_buffer_sgemm.cuh` — Dual buffer pipeline for compute/memory overlap
- [x] `tensor_core_sgemm.cuh` — WMMA API for mixed-precision FP16→FP32

**Deliverables**: Five kernel implementations in `src/kernels/`

---

### Phase 3: Utility Infrastructure ✅

**Completed Tasks**:
- [x] `cuda_utils.cuh` — RAII wrappers and exception-based error handling
- [x] `verify.cuh` — Correctness verification against cuBLAS reference
- [x] `benchmark.cuh` — Performance measurement framework using CUDA Events

**Deliverables**: Three utility modules in `src/utils/`

---

### Phase 4: Testing Suite ✅

**Completed Tasks**:
- [x] `test_sgemm.cu` — Google Test unit tests for all kernels
- [x] Property-based tests covering 100+ random dimension combinations
- [x] Tensor Core fallback tests for non-aligned dimensions
- [x] Edge case tests (1×1×1, unaligned sizes)

**Deliverables**: `tests/test_sgemm.cu`

---

### Phase 5: Build System & CI/CD ✅

**Completed Tasks**:
- [x] `CMakeLists.txt` — Primary CMake build system
- [x] `Makefile` — Quick local build alternative
- [x] `.github/workflows/ci.yml` — Format checks and containerized CUDA compile
- [x] `.github/workflows/pages.yml` — GitHub Pages deployment

**Deliverables**: `CMakeLists.txt`, `Makefile`, CI/CD workflows

---

### Phase 6: Documentation ✅

**Completed Tasks**:
- [x] `README.md` — English documentation
- [x] `README.zh-CN.md` — Chinese documentation
- [x] `CHANGELOG.md` — Version history following Keep a Changelog format
- [x] `index.md` — GitHub Pages landing page
- [x] `_config.yml` — Jekyll configuration

**Deliverables**: All documentation files at project root

---

### Phase 7: Code Quality & Refinement ✅

**Completed Tasks**:
- [x] **v2.0.0**: RAII refactoring across all kernels; exception-based error handling replacing `exit()` calls
- [x] **v2.1.0**: Dead code cleanup — removed 514 lines across 7 source files

**Deliverables**: Cleaner, more maintainable codebase

---

## Milestone Timeline

| Version | Date | Milestone | Key Changes |
|---------|------|-----------|-------------|
| 1.0.0 | 2025-02-13 | Project Initialization | MIT license, `.gitignore`, `.editorconfig`, basic README |
| 2.0.0-rc.1 | 2026-03-09 | Memory Leak Fixes | RAII refactoring, CMake build, self-contained project |
| 2.0.0-rc.2 | 2026-03-10 | GitHub Pages | Pages configuration, landing page, documentation enhancements |
| 2.0.0 | 2026-03-13 | Stable Release | CPU-safe CI workflow, format checks, containerized build validation |
| 2.1.0 | 2026-04-16 | Documentation & Code Cleanup | Dead code removal (514 lines), spec documentation reorganization |

---

## Constraints

### CON-ARCH-001: Supported GPU Architectures

| GPU | Architecture | Compute Capability | Build Flag |
|-----|-------------|-------------------|------------|
| Tesla V100 | Volta | sm_70 | `GPU_ARCH=sm_70` |
| RTX 2080 | Turing | sm_75 | `GPU_ARCH=sm_75` |
| RTX 3090 / A100 | Ampere | sm_80 / sm_86 | `GPU_ARCH=sm_86` |
| RTX 4090 / L40 | Ada Lovelace | sm_89 | `GPU_ARCH=sm_89` |
| H100 | Hopper | sm_90 | `GPU_ARCH=sm_90` |

### CON-ARCH-002: Build Commands

```bash
# CMake (recommended)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Makefile (quick local)
make GPU_ARCH=sm_86
make benchmark
make test
```

---

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [WMMA API Reference](https://docs.nvidia.com/cuda/cuda-c-api/group__CUDA__WMMA.html)
- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA's high-performance GEMM library
