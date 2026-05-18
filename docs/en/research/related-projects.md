---
title: Related Projects
---

# Related Projects

This page compares the whitepaper site and codebase against a few high-signal references. The goal is not to rank them. The goal is to clarify what this repository is trying to teach.

## Comparison matrix

| Project | What it is strongest at | What this repository borrows | What this repository does differently |
|---|---|---|---|
| CUTLASS | Production-grade GEMM building blocks, template depth, architecture specialization | Respect for explicit tile hierarchy and Tensor Core constraints | Keeps the ladder readable, smaller, and interview-friendly instead of pushing toward industrial abstraction |
| NVIDIA CUDA Samples | Minimal official examples that clarify APIs and execution concepts | Canonical API usage patterns and baseline correctness expectations | Adds a staged narrative, validation framing, and comparative interpretation |
| Si Bohm SGEMM worklog | Public optimization diary with strong pedagogical value | The idea that every speedup should be tied to one bottleneck shift | Adds bilingual documentation, explicit validation boundaries, and a more curated research surface |
| BLIS and CPU GEMM literature | Layered thinking about blocking, packing, and hierarchy | Performance reasoning through memory movement and reuse | Stays GPU-specific, with WMMA guards and CUDA runtime constraints in focus |

## How to use these references well

1. Use CUTLASS when you want to see where maintainable high-performance CUDA GEMM gets significantly more abstract.
2. Use NVIDIA samples when you want the smallest official demonstration of an API or Tensor Core feature.
3. Use public SGEMM optimization diaries when you want to compare explanatory order and trade-off framing.

## What not to confuse

- This repository is not trying to replace cuBLAS or CUTLASS.
- It is not trying to be the smallest CUDA sample.
- It is not a benchmark-only notebook.

Its value is the combination: readable ladder, explicit trust boundary, and a public narrative that can survive technical questioning.
