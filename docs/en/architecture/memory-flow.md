---
title: Memory Flow
---

# Memory Flow

This page treats SGEMM as a data-movement system: the kernels differ mainly in how they move matrix tiles through global memory, shared memory, registers, and specialized compute units.

## End-to-end data path

```text
Global memory (A, B) 
    ↓
Coalesced block loads
    ↓
Shared-memory staging
    ↓
Register or WMMA-fragment consumption
    ↓
FP32 accumulation
    ↓
Global-memory store to C
```

The architectural goal is to make each step cheaper, more reusable, or better scheduled.

## System view by stage

| Stage | What changes in the flow | Why it matters |
|------|---------------------------|----------------|
| Naïve | Threads read directly from global memory for every multiply-accumulate step | Minimal reuse; poor locality on matrix B |
| Tiled | Blocks stage tiles of A and B in shared memory | Global loads are amortized across many operations |
| Bank-Free | Shared tiles are padded to change bank mapping | Shared-memory bandwidth becomes more predictable |
| Double Buffer | Two shared-memory tile slots alternate between load and compute | Part of the load latency can be hidden behind math |
| Tensor Core | FP16 tiles feed WMMA fragments, then accumulate in FP32 | Throughput rises, but staging and guards become stricter |

## 1. Global-memory behavior: the first problem

In the naïve kernel, every output element is computed with direct reads from global memory. That exposes two expensive realities:

- one row of **A** is effectively revisited many times
- one column of **B** becomes a stride-heavy access pattern across memory

The first architectural move is therefore not “more math,” but “less wasteful movement.”

## 2. Shared-memory staging: the first major redesign

Tiling introduces a reusable working set inside each thread block.

### What changes

- the block cooperatively loads a tile of **A**
- the block cooperatively loads a tile of **B**
- all threads reuse those tiles before the next stage is fetched

### What new responsibility appears

Shared memory is fast only when the block respects its coordination rules:

- loads must finish before compute starts
- compute must finish before the next tile overwrites the buffer
- tile size affects occupancy and shared-memory footprint

This is why the tiled kernel introduces `__syncthreads()` as a correctness boundary, not just a performance detail.

## 3. Bank conflicts: when “shared memory” is still not enough

Moving data into shared memory does not automatically make access efficient. If many threads hit the same bank pattern, the accesses serialize.

### Repository strategy

The bank-free kernel pads the shared-memory leading dimension from `TILE_SIZE` to `TILE_SIZE + 1`.

### System effect

- the logical algorithm stays the same
- the physical address stride changes
- the common conflict-heavy pattern is broken

That is a memory-system decision, not a numerical one. It exists to make the staged data easier for the SM to serve at warp speed.

## 4. Buffer scheduling: from storage to pipeline

Double buffering changes memory flow from a single staging slot to a pipeline.

```text
Iteration t:     compute on buffer 0   | preload buffer 1
Iteration t + 1: compute on buffer 1   | preload buffer 0
```

The value of this step is not extra reuse. It is overlap:

- current data remains available for computation
- next data begins moving early
- the tile loop becomes closer to a producer/consumer schedule

This is why double buffering belongs in architecture discussion even if its speedup is smaller than the jump from naïve to tiled.

## 5. Tensor Core staging: stricter flow for a faster unit

Tensor Core execution adds another staging rule:

- inputs must be prepared as FP16
- dimensions must satisfy WMMA tile alignment
- work is consumed as warp-level fragments rather than scalar register loops

The output is still accumulated into FP32, which is why the repository describes the path as mixed precision rather than pure FP16.

Read [Tensor Core Path](/en/architecture/tensor-core-path) for the decision logic behind those guards.

## Memory-flow design principles

### Coalescing before cleverness

The architecture prioritizes making global-memory traffic sane before adding more exotic optimizations.

### Reuse before peak throughput

Shared-memory tiling appears before Tensor Core acceleration because a clear reuse story is easier to validate and explain than a hardware-specific fast path.

### Predictability before hero numbers

Padding away bank conflicts and keeping fallback logic explicit are both examples of the same value: the repository prefers explainable behavior over fragile “best-case only” claims.

## Deep links

- [Tiled Kernel](/en/academy/kernel-tiled)
- [Bank Conflict Free](/en/academy/kernel-bank-free)
- [Double Buffer](/en/academy/kernel-double-buffer)
- [Tensor Core WMMA](/en/academy/kernel-tensor-core)
- [CUDA Memory Cheat Sheet](/en/academy/cuda-memory-cheatsheet)
- [Resources Hub](/en/research/)
