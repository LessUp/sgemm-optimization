---
title: System Blueprint
---

# System Blueprint

This page is a full component-level blueprint of the SGEMM optimization system. It maps every major component, the data flows between them, and the design decisions that constrain those flows.

<ThemedFigure
  :wide="true"
  light="/figures/whitepaper-system-light.svg"
  dark="/figures/whitepaper-system-dark.svg"
  alt="Full system blueprint showing kernel components, data paths, and the validation and research rails attached to the ladder."
  caption="The blueprint maps every component that has an explicit architectural reason to exist. Components without a reason are either not here or marked as open questions."
/>

## Component inventory

| Component | Role | Constraints |
|---|---|---|
| `main.cu` entry | Parses arguments, selects kernel variant, routes to benchmarking or correctness check | Must not hard-code a variant; selection is runtime |
| `kernels/sgemm_naive.cuh` | Baseline FP32, one thread per output element | Establishes the cost model; no shared memory |
| `kernels/sgemm_tiled.cuh` | Tiled FP32 with shared-memory staging | Tile size is a compile-time template parameter |
| `kernels/sgemm_bank_free.cuh` | Tiled FP32 with padding to eliminate bank conflicts | Padding is the only structural difference from tiled |
| `kernels/sgemm_double_buffer.cuh` | Overlapped staging and compute using double buffering | Requires at least two staging buffers in shared memory |
| `kernels/sgemm_tensor_core.cuh` | WMMA-based computation with FP32 entry path | Guarded by device capability and shape divisibility |
| `utils/cuda_check.h` | RAII-based error propagation for CUDA API and kernel calls | Throws `std::runtime_error` on failure; no silent returns |
| `utils/matrix.h` | Host-side matrix initialization and reference computation | Row-major layout; reference uses CPU BLAS or naive FP64 loop |
| `tests/test_sgemm.cu` | cuBLAS-backed oracle correctness suite | Runs only on GPU; not included in hosted CI |
| Docs site | Narrative layer — architecture, academy, validation, research | VitePress with bilingual routes; no runtime GPU dependency |

## Data flow: host to device

```
Host allocates A, B, C (row-major, FP32)
  │
  ▼
cudaMemcpy H→D
  │
  ▼
Kernel launch (grid, block, shared-memory budget)
  ├─ Naive path: direct global reads per thread
  ├─ Tiled path: cooperative staging into shared tile
  ├─ Bank-free path: padded tile staging
  ├─ Double-buffer path: async prefetch of next tile
  └─ Tensor Core path: FP32→FP16 conversion + WMMA fragment accumulation
  │
  ▼
cudaMemcpy D→H
  │
  ▼
Correctness check against cuBLAS oracle (local GPU only)
```

## Design decisions and their architectural consequences

### RAII error handling

All CUDA API calls and kernel launches are wrapped in `cudaCheck()`. This ensures that any failure path immediately terminates with a traceable error rather than silently propagating incorrect results through the pipeline.

**Consequence:** Test code cannot accidentally swallow an error and then compare incorrect output against the cuBLAS oracle, which would make a failing kernel appear to pass.

### Runtime kernel selection

The entry point selects the kernel variant at runtime from a command-line argument, rather than compiling multiple executables.

**Consequence:** Benchmark comparisons between variants use the same binary and the same host code path, making the comparison cleaner and eliminating build-flag confounds.

### Template tile sizes

Tile dimensions are compile-time template parameters, not runtime constants.

**Consequence:** The shared-memory layout is known at compile time, enabling the compiler to generate efficient addressing and avoiding dynamic shared-memory allocation overhead. The tradeoff is that only the compiled tile sizes can be benchmarked without a rebuild.

### Tensor Core as guarded optional path

The Tensor Core variant checks device capability and shape divisibility before committing to WMMA computation, and falls back to the FP32 tiled path otherwise.

**Consequence:** The system is safe to run on non-Tensor-Core hardware, and benchmark results from such hardware are labeled as FP32 results, not mixed-precision results.

## Validation boundary in the blueprint

The blueprint explicitly separates compile-time-verifiable invariants from runtime-verifiable invariants:

| Invariant class | Verifiable where |
|---|---|
| File structure, docs, OpenSpec alignment | Hosted CI |
| CUDA code compiles without warning | Hosted CI (compile-only) |
| Correctness under cuBLAS oracle | Local GPU run |
| Benchmark numbers and speedup ratios | Local GPU run with named hardware |

## Related pages

- [Architecture Overview](./index) — system map and design rationale
- [Kernel Ladder](./kernel-ladder) — optimization order and bottleneck shifts
- [Tensor Core Path](./tensor-core-path) — Tensor Core constraints and fallback behavior
- [Correctness Policy](../validation/correctness-policy) — oracle definition and tolerance thresholds
- [Performance Model](../validation/performance-model) — quantitative cost model
