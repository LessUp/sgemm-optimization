---
title: Kernel Ladder
---

# Kernel Ladder

The kernel ladder is the repository's main reasoning chain. Each rung exists because the previous one exposed a bottleneck that the next one addresses.

## The ladder at a glance

| Stage | Main bottleneck exposed | Architectural move | Trade-off | Deep dive |
|------|--------------------------|--------------------|-----------|-----------|
| Naïve | Repeated global-memory traffic, poor reuse | One thread computes one output element | Readable but memory-bound | [Naïve Kernel](/en/academy/kernel-naive) |
| Tiled | Global-memory bandwidth pressure | Shared-memory tiling and block-level reuse | Requires barriers and tile-size choices | [Tiled Kernel](/en/academy/kernel-tiled) |
| Bank-Free | Shared-memory contention | Pad tiles to remove bank-conflict patterns | Slightly higher shared-memory footprint | [Bank Conflict Free](/en/academy/kernel-bank-free) |
| Double Buffer | Load/compute serialization | Ping-pong shared-memory buffers | More shared memory and more scheduling complexity | [Double Buffer](/en/academy/kernel-double-buffer) |
| Tensor Core | CUDA-core throughput ceiling | Warp-level WMMA on Tensor Cores | FP16 staging, shape guards, and fallback policy | [Tensor Core WMMA](/en/academy/kernel-tensor-core) |

## Why the ladder matters

Many SGEMM demos present multiple kernels as separate tricks. This repository keeps them in one narrative so the reader can answer four questions at every step:

1. **What bottleneck was visible?**
2. **What architectural mechanism addresses it?**
3. **What new cost or constraint appears?**
4. **How should that change be validated?**

That makes the ladder useful for both technical study and interview storytelling.

## Rung-by-rung reasoning chain

### 1. Naïve: make the bottleneck obvious

The naïve kernel is intentionally simple. It gives every thread one output element and performs a straightforward inner product. That simplicity makes the memory problem visible:

- rows of **A** are reread many times
- columns of **B** are accessed with poor global-memory locality
- arithmetic work is easy to describe, but data reuse is weak

Use it as the baseline for explaining why SGEMM is often limited by data movement before clever compute scheduling even begins.

### 2. Tiled: buy reuse with cooperation

Tiling changes the story from “each thread fetches what it needs” to “the block stages a reusable working set in shared memory.”

The architectural effect is larger than the code change suggests:

- global-memory accesses become more coalesced
- the same staged tile is reused across many multiply-accumulate steps
- synchronization becomes a real correctness requirement

Tiling is the first step where the system starts looking like GPU architecture rather than plain matrix math.

### 3. Bank-Free: fix the shared-memory shape, not just the algorithm

Once shared memory becomes central, its layout matters. The bank-free kernel exists because a tiled kernel can still waste cycles when many threads hit the same bank pattern.

Padding the shared tile changes the address stride so that the common access pattern no longer collapses into avoidable conflicts. Even when the sample benchmark does not show a dramatic speedup over tiled, the reasoning still matters: the layout is more robust and easier to defend as a system design choice.

### 4. Double Buffer: turn the tile loop into a schedule

After shared-memory reuse and bank layout are under control, the next question is timing: can the next tile be staged while the current one is still being consumed?

Double buffering answers that by turning one tile buffer into two alternating roles:

- **current buffer** feeds compute
- **next buffer** is loaded for the following iteration

This is the point where the ladder becomes a scheduling story, not only a memory-layout story.

### 5. Tensor Core: add a guarded fast path

Tensor Core acceleration is intentionally the last rung because it depends on the earlier reasoning:

- readers already understand why data movement matters
- the project can compare WMMA results against strong FP32 baselines
- the implementation can explain why a faster path must still be guarded

The repository treats WMMA as a constrained acceleration path, not a universal replacement. Friendly shapes can use compute-only WMMA benchmarking; unsupported shapes must stay honest and fall back.

## What the ladder is trying to teach

| Lesson | Where it appears |
|--------|------------------|
| Global-memory access pattern dominates first-order behavior | Naïve → Tiled |
| Shared memory is useful only when its layout is also correct | Tiled → Bank-Free |
| Scheduling and overlap matter after locality is improved | Bank-Free → Double Buffer |
| Higher-throughput hardware introduces API and shape constraints | Double Buffer → Tensor Core |
| Correctness, measurement scope, and fallback rules belong in the design narrative | Whole ladder |

## How to talk about performance without oversimplifying

The ladder is a reasoning chain, not a promise of perfectly monotonic benchmark numbers on every machine. For example:

- tiled and bank-free may trade places depending on occupancy and access details
- double buffering can bring only modest gains on GPUs that already hide latency well
- Tensor Core numbers must be split into end-to-end and compute-only views

The right question is not “which page has the biggest number?” but “which bottleneck does this step address, and under what assumptions?”

## Related pages

- [Architecture Overview](/en/architecture/)
- [Memory Flow](/en/architecture/memory-flow)
- [Tensor Core Path](/en/architecture/tensor-core-path)
- [Benchmark Scope](/en/validation/benchmark-scope)
