# Testing Specification

> **Version**: 2.1.0 | **Last Updated**: 2026-04-23 | **Status**: Complete

## Purpose

Define BDD-style test specifications for SGEMM kernel verification against the cuBLAS reference.

## Requirements

### Requirement: Test Framework
The project SHALL use Google Test for unit testing with CUDA support.

#### Scenario: Tests execute via CTest
- **WHEN** a developer runs `ctest --test-dir build`
- **THEN** all kernel tests SHALL execute and report results

### Requirement: Correctness Verification
All kernels SHALL be verified against cuBLAS reference output.

#### Scenario: Standard kernels pass correctness tests
- **WHEN** standard FP32 kernels are tested
- **THEN** they SHALL match cuBLAS within rtol=1e-3, atol=1e-4

#### Scenario: Tensor Core passes correctness tests
- **WHEN** Tensor Core kernel is tested with aligned dimensions
- **THEN** it SHALL match cuBLAS within rtol=5e-2, atol=1e-2

#### Scenario: Tensor Core handles unaligned dimensions
- **WHEN** Tensor Core kernel receives non-16-aligned dimensions
- **THEN** it SHALL safely fall back to FP32 implementation

### Requirement: Repository validation includes governance integrity
The repository test and validation model MUST cover not only code correctness but also the integrity of governance, specification, and documentation structure.

#### Scenario: Governance-related files change
- **WHEN** OpenSpec files, workflow files, governance documents, or documentation structure are modified
- **THEN** the repository MUST provide CI-safe validation that checks the relevant specification and structural invariants before those changes are treated as complete

### Requirement: Validation expectations are split by execution environment
The repository MUST document which checks are expected to run in hosted CI and which checks require a local GPU-capable environment.

#### Scenario: Contributor reads validation guidance
- **WHEN** a contributor prepares to validate repository changes
- **THEN** the documented workflow MUST separate CI-safe checks such as formatting, compilation, OpenSpec validation, or Pages buildability from GPU-required runtime verification and benchmarking

---

## Test Scenarios

### Scenario 1: Standard Kernel Correctness

**Given** a square matrix dimension (e.g., 512×512×512, 1024×1024×1024)
**When** each standard kernel (Naive, Tiled, Bank-Free, Double-Buffer) is executed
**Then** the output matrix C must match cuBLAS result within tolerance `rtol=1e-3, atol=1e-4`

**Test Cases**:
- Aligned square dimensions: 512, 1024
- Non-square dimension: 256×384×640
- Random dimensions: 100+ property-based test iterations

---

### Scenario 2: Tensor Core Correctness

**Given** matrix dimensions that are multiples of 16 (WMMA alignment requirement)
**When** the Tensor Core kernel is executed with FP16→FP32 mixed precision
**Then** the output matrix C must match cuBLAS result within tolerance `rtol=5e-2, atol=1e-2`

**Test Cases**:
- 16-aligned dimensions: 512×512×512, 1024×1024×1024
- Edge alignment: 16×16×16, 32×32×32

---

### Scenario 3: Tensor Core Fallback

**Given** matrix dimensions that are NOT multiples of 16
**When** the Tensor Core kernel is invoked
**Then** the kernel must safely fall back to an FP32 implementation without errors

**Test Cases**:
- Unaligned dimension: 511×513×1025
- Prime dimensions: 17×19×23
- Small unaligned: 15×15×15

---

### Scenario 4: Edge Cases

**Given** minimal or extreme matrix dimensions
**When** any kernel is executed
**Then** the kernel must complete without CUDA errors and produce correct results

**Test Cases**:
- Minimal: 1×1×1
- Small: 2×2×2, 4×4×4
- Non-square edge: 1×128×1, 128×1×128, 1×128×128

---

### Scenario 5: Performance Benchmarking

**Given** a set of benchmark dimensions
**When** each kernel is benchmarked using CUDA Events
**Then** the GFLOPS must be reported and compared against cuBLAS baseline

**Benchmark Dimensions**:
- 512×512×512
- 1024×1024×1024
- 2048×2048×2048
- 256×384×640 (non-square)
- 511×513×1025 (unaligned)

**Expected Performance Progression**:
```
Naive < Tiled < Bank-Free < Double-Buffer < Tensor Core << cuBLAS
```

---

## Property-Based Testing

### Property 1: Numerical Correctness

**For all** valid matrix dimensions M, K, N where 1 ≤ M, K, N ≤ 2048:
**Verify** that `allclose(kernel_output, cublas_output, rtol, atol)` returns true

**Sampling Strategy**:
- 100+ random dimension triples
- Include squares, non-squares, and edge cases

---

### Property 2: Kernel Monotonicity (Performance)

**For all** benchmark dimensions:
**Verify** that each successive optimization stage achieves equal or better GFLOPS than the previous stage

**Expected Order**:
```
GFLOPS(Naive) ≤ GFLOPS(Tiled) ≤ GFLOPS(Bank-Free) ≤ GFLOPS(Double-Buffer) ≤ GFLOPS(Tensor Core)
```

---

### Property 3: Memory Safety

**For all** kernel executions:
**Verify** that:
- No CUDA errors occur
- No memory leaks (RAII validation)
- All GPU resources are properly released

---

## Tolerance Specifications

| Kernel Category | Relative Tolerance (rtol) | Absolute Tolerance (atol) | Precision |
|-----------------|--------------------------|--------------------------|-----------|
| Standard FP32 Kernels (Naive, Tiled, Bank-Free, Double-Buffer) | 1e-3 | 1e-4 | FP32 |
| Tensor Core (FP16→FP32 mixed precision) | 5e-2 | 1e-2 | FP16 input, FP32 accumulate |

---

## Test Execution Commands

```bash
# Build and run tests via CMake
cmake --build build --target test_sgemm
ctest --test-dir build --verbose

# Build and run tests via Make
make test

# Run specific test cases
./build/bin/test_sgemm --gtest_filter="*Correctness*"
./build/bin/test_sgemm --gtest_filter="*TensorCore*"
./build/bin/test_sgemm --gtest_filter="*EdgeCase*"
```

---

## Acceptance Criteria

All tests must pass for the following conditions:

1. **Correctness**: All kernels match cuBLAS within specified tolerances
2. **Fallback**: Tensor Core safely handles unaligned dimensions
3. **Edge Cases**: All minimal and extreme dimensions complete successfully
4. **Performance**: Performance progression is monotonic (non-regressing)
5. **Memory Safety**: No leaks, no errors, clean resource management

---

## References

- [Google Test Documentation](https://google.github.io/googletest/)
- [CUDA Testing Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- Property-based testing inspired by [Hypothesis](https://hypothesis.works/) methodology
