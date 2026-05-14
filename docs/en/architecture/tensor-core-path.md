---
title: Tensor Core Path
---

# Tensor Core Path

Tensor Core support is treated as a guarded fast path. The repository does not assume every matrix shape, every GPU, or every benchmark label should go through WMMA.

## Two public paths

| Path | Input type | Intended use | Behavior when unsupported |
|------|------------|--------------|---------------------------|
| `launch_tensor_core_sgemm_with_fallback` | FP32 | Safe end-to-end execution and public benchmarking | Falls back to an explicit FP32 kernel |
| `launch_tensor_core_sgemm_fp16` | FP16 | Compute-only WMMA benchmarking | Throws when device or dimensions do not satisfy WMMA requirements |

This split is central to the architecture story because it prevents conversion cost and fallback behavior from being misreported as pure Tensor Core compute speed.

## Path-selection logic

### 1. Device guard

WMMA requires Tensor Core support (`sm_70+`). If the device does not provide that capability, the safe FP32 wrapper stays on the fallback path.

### 2. Shape guard

The WMMA path expects dimensions aligned to the fragment shape. In this repository that means `M`, `K`, and `N` must be multiples of 16 for the fast path.

### 3. Data-format guard

The safe public entry point begins with FP32 inputs, so it must allocate FP16 staging buffers and convert both input matrices before launching the FP16 WMMA kernel.

### 4. Fallback policy

When the guard checks fail, the repository does not try to fake Tensor Core execution. It calls an explicit fallback chosen by the caller. In the repository's benchmark and recommended helper path, that fallback is the bank-conflict-free FP32 kernel.

## Why the fallback matters

The fallback is not a minor implementation detail. It makes three architectural promises true:

- **Correctness remains available for irregular shapes**.
- **Benchmark labels stay honest** because unsupported cases are not counted as WMMA wins.
- **The Tensor Core module stays modular** by requiring the caller to think about unsupported conditions.

## Guarded flow in practice

```text
FP32 inputs
  ↓
Check device capability and 16-aligned dimensions
  ├─ unsupported → explicit FP32 fallback
  └─ supported
       ↓
    Convert A and B from FP32 to FP16
       ↓
    Launch WMMA fast path
       ↓
    Accumulate into FP32 output C
```

The compute-only benchmark starts later in that flow: it assumes preconverted FP16 inputs and skips both conversion and fallback.

## Shape constraints and reporting discipline

| Question | Architecture answer |
|----------|---------------------|
| Can every matrix use WMMA? | No. Friendly dimensions are required. |
| Are end-to-end and compute-only numbers interchangeable? | No. One includes conversion/fallback behavior; the other isolates WMMA compute. |
| Does Tensor Core replace FP32 kernels? | No. It augments them with a constrained fast path. |
| What fallback does the repository usually wire in? | The bank-conflict-free FP32 kernel helper. |

## Why WMMA is still worth including

Even though the implementation is educational rather than cuBLAS-class, the Tensor Core path adds an important final architectural lesson:

- specialized hardware can raise the throughput ceiling
- higher throughput comes with shape, API, and precision constraints
- a trustworthy system must explain when it uses the fast path and when it declines to

## Read alongside

- [Architecture Overview](/en/architecture/)
- [Kernel Ladder](/en/architecture/kernel-ladder)
- [Benchmark Results](/en/benchmark-results)
- [Tensor Core WMMA](/en/kernel-tensor-core)
