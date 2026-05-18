---
title: Architecture Overview
---

# Architecture Overview

This section is the normative map of the SGEMM system. It records: what each component does, how data moves through the memory hierarchy, when kernel selection logic fires, and where the system's invariants and boundaries lie. Read it before trusting any benchmark claim or interview statement about this project.

<ThemedFigure
  :wide="true"
  light="/figures/kernel-ladder-light.svg"
  dark="/figures/kernel-ladder-dark.svg"
  alt="Kernel ladder diagram showing naive FP32, tiled FP32, bank-free FP32, double buffer, and Tensor Core WMMA with attached architecture, validation, and research rails."
  caption="The ladder is a map of bottleneck shifts, not a trophy rack. Each stage exists because the previous one exposed a new limit."
/>

## Technical thesis

SGEMM optimization on modern NVIDIA hardware is a sequence of bottleneck-class migrations. Moving from naïve FP32 to Tensor Core WMMA requires solving four distinct problems in order: DRAM saturation, shared-memory bank conflicts, staging–compute overlap, and WMMA hardware constraints. This repository structures its kernel implementations to surface one problem per stage, hold the prior stages fixed, and make the performance effect of each architectural decision independently observable.

## Component inventory

| Component | Layer | Primary responsibility |
|-----------|-------|------------------------|
| `src/main.cu` | Driver | Dispatch, size validation, timing harness, output |
| `src/kernels/naive.cuh` | Kernel | FP32 baseline with full global-memory load cost |
| `src/kernels/tiled.cuh` | Kernel | Cooperative tile load into shared memory; SMEM reuse |
| `src/kernels/bank_free.cuh` | Kernel | Padding eliminates shared-memory bank conflicts |
| `src/kernels/double_buffer.cuh` | Kernel | Async prefetch overlaps next-tile staging with active compute |
| `src/kernels/tensor_core.cuh` | Kernel | WMMA fragment accumulation on hardware-aligned tiles |
| `src/utils/cuda_check.cuh` | Utility | CUDA error checking and RAII device-resource guards |
| `tests/test_sgemm.cu` | Test | Correctness verification under reference tolerance |

## Memory-hierarchy data flow

Each kernel optimization step corresponds to a change in which memory level dominates access cost:

```
Naïve:       [Global memory]  → registers         (DRAM-bound)
Tiled:       [Global memory]  → SMEM → registers  (SMEM reuse, conflict risk)
Bank-Free:   [Global memory]  → SMEM+pad → regs   (conflict-free, latency exposed)
Dbl-Buffer:  [Global memory]  → double-SMEM → regs (staging hidden behind compute)
Tensor Core: [Global memory]  → SMEM → WMMA frags  (hardware-accelerated accumulation)
```

The memory-flow page makes this concrete: addresses, strides, tile dimensions, and the exact load patterns used at each stage.

## Design invariants

These properties are held constant across all kernel stages and are part of the architecture's correctness contract:

1. **Row-major layout throughout.** All matrices A, B, and C use row-major storage. No kernel silently assumes column-major order.
2. **Float4 granularity for vectorized loads.** Kernels that benefit from wider loads use `float4` to maximize per-instruction memory bandwidth.
3. **Fallback on unsatisfied constraints.** The Tensor Core entry path falls back to the FP32 path when shape guards are not met. Benchmark numbers are never reported from a fallback-activated run.
4. **Epsilon-bounded correctness.** Test harnesses verify outputs against a cuBLAS reference with a per-element tolerance of `1e-3`. Kernel correctness is not assumed; it is measured.
5. **Timing outside CUDA graph bounds.** Benchmark timing wraps the full device call including synchronization. Cold-start and warm-up behavior is documented per benchmark result.

## Kernel selection and fallback logic

The entry path in `src/main.cu` selects a kernel tier based on device capability queries and matrix dimension checks:

| Condition | Path selected |
|-----------|---------------|
| Any GPU, any shape | FP32 ladder (naïve → double buffer) |
| SM ≥ 7.0, shape divisible by WMMA tile | Tensor Core WMMA path |
| SM ≥ 7.0, shape not WMMA-aligned | FP32 path (fallback) |
| SM < 7.0 | FP32 path (fallback) |

The pure benchmark invokes the Tensor Core kernel directly on a pre-validated shape. The safe entry path uses runtime guards.

## Architectural decisions

### 1. The ladder, not the bag of tricks

Each kernel solves one bottleneck class:

1. **Naïve** establishes the arithmetic-intensity bound and exposes DRAM saturation.
2. **Tiled** moves data into shared memory cooperatively, exposing bank conflict as the next limit.
3. **Bank-Free** pads shared-memory arrays to remove conflict, exposing staging latency.
4. **Double Buffer** overlaps next-tile staging with active compute to reduce stall cycles.
5. **Tensor Core** uses hardware-fused matrix accumulation under strict alignment and device constraints.

The point is that each stage has a single reason to exist and a single architectural effect that can be measured.

### 2. Validation as an architectural first class

The project separates what two different environments can prove:

| Claim | Local CUDA GPU | Hosted CI |
|-------|----------------|-----------|
| Compilation succeeds | ✓ | ✓ (structure check only) |
| Output correctness vs. cuBLAS | ✓ | ✗ |
| Benchmark performance claims | ✓ | ✗ |
| Repository structure and docs | ✓ | ✓ |
| VitePress Pages buildability | ✓ | ✓ |

This is not just process hygiene. It affects which claims a reader can trust from CI green status alone.

### 3. Tensor Core as an explicit fast path

The FP32 ladder and the Tensor Core path are independent tiers. The repository exposes both so that:

- Benchmark claims for WMMA are only made on aligned shapes with SM ≥ 7.0 devices.
- The FP32 ladder remains a complete, self-contained teaching path that does not require Tensor Core hardware.
- Fallback behavior is tested, not assumed.

## System map and reading paths

| Need | Go to |
|------|-------|
| Full component and data-flow diagram | [System Blueprint](./system-blueprint) |
| Kernel-by-kernel explanation of bottleneck shifts | [Kernel Ladder](./kernel-ladder) |
| Memory hierarchy and load-pattern analysis | [Memory Flow](./memory-flow) |
| WMMA selection, shape guards, fallback | [Tensor Core Path](./tensor-core-path) |
| Ordered teaching path, interview framing | [Academy](../academy/) |
| Correctness policy and benchmark scope | [Validation](../validation/) |
| External references and comparisons | [Research](../research/) |

## Fast reviewer path

1. This page: architectural thesis and invariants.
2. [System Blueprint](./system-blueprint): full component inventory with data flow.
3. [Validation Overview](../validation/): what the evidence can and cannot prove.
4. [Benchmark Results](../validation/benchmark-results): numbers with scope attached.
5. [Academy](../academy/): the ordered explanation for interview defense.
