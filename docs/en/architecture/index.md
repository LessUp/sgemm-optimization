---
title: Architecture Overview
---

# Architecture Overview

This section is the canonical map of the SGEMM system: why the design exists, how data moves, when each kernel strategy appears, and where Tensor Core acceleration is allowed to take over.

## Why this design exists

This repository is not organized as “one fast kernel plus a benchmark screenshot.” It is organized as an engineering reasoning chain:

- start from a readable FP32 baseline
- expose the next bottleneck instead of hiding it
- add one architectural idea at a time
- keep correctness and benchmark scope explicit
- preserve a safe path when Tensor Core constraints are not met

That structure makes the project useful for learning, review, and interviews: readers can explain **why** a kernel exists before they talk about how fast it is.

## What this repository is trying to prove

- SGEMM optimization should read as a reasoning chain, not a bag of isolated tricks.
- Performance claims only count when they stay attached to correctness policy and benchmark scope.
- Tensor Core acceleration is only persuasive when constraints and fallback behavior are explicit.

## System map

| Layer | Responsibility | Where to go next |
|------|----------------|------------------|
| Kernel ladder | Explains the optimization chain from naïve FP32 to WMMA | [Kernel Ladder](/en/architecture/kernel-ladder) |
| Memory flow | Explains global-memory access, shared-memory reuse, bank conflicts, and double buffering as one data-movement story | [Memory Flow](/en/architecture/memory-flow) |
| Tensor Core path | Explains WMMA selection, FP32→FP16 staging, shape guards, and fallback behavior | [Tensor Core Path](/en/architecture/tensor-core-path) |
| Deep kernel pages | Explains each kernel implementation in isolation | [Naïve](/en/kernel-naive), [Tiled](/en/kernel-tiled), [Bank-Free](/en/kernel-bank-free), [Double Buffer](/en/kernel-double-buffer), [Tensor Core WMMA](/en/kernel-tensor-core) |

## Architectural decisions that shape the repository

### 1. Optimization is presented as a ladder, not a bag of tricks

Each kernel solves a specific bottleneck class:

1. **Naïve** establishes the cost model and exposes poor reuse.
2. **Tiled** trades extra coordination for shared-memory reuse.
3. **Bank-Free** stabilizes shared-memory access by padding away avoidable conflicts.
4. **Double Buffer** overlaps staging and compute to hide part of memory latency.
5. **Tensor Core** raises the throughput ceiling, but only under explicit device and shape constraints.

The goal is not “every later kernel must beat every earlier kernel on every GPU.” The goal is that each step has a clear reason to exist and a measurable architectural effect.

### 2. Data movement is the main system story

SGEMM performance here is framed around where data lives and when it moves:

- from global memory into the SM
- from global memory into shared tiles
- from shared memory into registers or WMMA fragments
- from staged tiles back into output matrix C

That is why the architecture section treats memory flow as a first-class topic instead of leaving it scattered across per-kernel notes.

### 3. Tensor Core is an optional fast path, not the only path

The repository exposes both:

- a **safe FP32 entry path** that may convert inputs and fall back when WMMA is unsupported
- a **pure compute-only WMMA path** used to measure raw Tensor Core behavior under friendly shapes

This keeps benchmark claims honest. Unsupported dimensions are not silently reported as Tensor Core wins.

### 4. Validation boundaries are part of the architecture

The project deliberately separates what can be trusted in different environments:

| Area | Local CUDA GPU | Hosted CI |
|------|----------------|-----------|
| CUDA compilation | Yes | No |
| Runtime correctness | Yes | No |
| Benchmark performance | Yes | No |
| Docs, OpenSpec, and repository integrity | Yes | Yes |
| Pages buildability | Optional | Yes |

This is not just process documentation. It affects how the architecture is narrated: performance conclusions only count when they are tied back to the correct runtime environment.

## Recommended reading path

1. Start here for the system map.
2. Read [Kernel Ladder](/en/architecture/kernel-ladder) to understand the optimization chain.
3. Read [Memory Flow](/en/architecture/memory-flow) to understand the data-movement logic behind the ladder.
4. Read [Tensor Core Path](/en/architecture/tensor-core-path) before interpreting WMMA benchmark numbers.
5. Use the existing kernel deep dives when you want implementation detail instead of system rationale.

## Fast reviewer path

1. Read this page for the system claim.
2. Read [Kernel Ladder](/en/architecture/kernel-ladder) for the optimization order.
3. Read [Validation Overview](/en/validation/) before trusting any benchmark claim.
4. Read [Methodology](/en/methodology/) when you need the concise explanation path used in reviews or interviews.
5. Use [Resources Hub](/en/resources/) to trace external sources and comparison points.

## Related resources

- [Resources Hub](/en/resources/)
- [Validation Overview](/en/validation/)
- [Learning Path](/en/learning-path)
- [Getting Started](/en/getting-started)
- [Stable architecture spec](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/specs/architecture/spec.md)
