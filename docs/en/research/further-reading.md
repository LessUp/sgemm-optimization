---
title: Further Reading Routes
---

# Further Reading Routes

This page is intentionally opinionated. It is here to answer "what should I study next, and why?" without making you decode a random list of links.

## GEMM tiling and hierarchy thinking

Study this route when tiled SGEMM makes sense mechanically, but the bigger design logic still feels fuzzy.

- Revisit [Tiled Kernel](/en/academy/kernel-tiled) and [Memory Flow](/en/architecture/memory-flow) first so the project-specific examples are fresh.
- Then read [Anatomy of High-Performance Matrix Multiplication](/en/research/references#papers-and-performance-mental-models) to see why blocking and data reuse define the problem at a systems level.
- Compare that model with [CUTLASS](/en/research/references#exemplary-codebases-and-production-grade-samples) to see how a production CUDA library expresses hierarchy and tiling.

Questions to keep in mind:

- Which memory level is each tile protecting?
- Which part of the design reduces bandwidth pressure versus launch overhead?
- What changed between the teaching kernel and a production template stack?

## Occupancy as a constraint, not a vanity metric

Study this route when you keep hearing "occupancy" but cannot tell whether it is the cause of a slowdown or just a correlated number.

- Start with [Double Buffer](/en/academy/kernel-double-buffer) and [Bank Conflict Free](/en/academy/kernel-bank-free), because register pressure and shared-memory layout are the usual reasons occupancy changes.
- Use the [CUDA Occupancy Calculator and Nsight Compute references](/en/research/references#profiler-tooling-and-diagnosis-references) to connect launch configuration choices to active warps, shared-memory usage, and register limits.
- Then reread [Diagnosis Loop](/en/academy/diagnosis-loop) so occupancy becomes one hypothesis to test, not the answer to every performance question.

Questions to keep in mind:

- Did occupancy drop because the kernel got worse, or because it now does more useful work per block?
- Which resource is binding first: registers, shared memory, or block size?
- What profiler metric would falsify your current story?

## Roofline thinking for SGEMM

Study this route when you want a better language for "memory-bound" versus "compute-bound" than intuition alone.

- Use [Benchmark Discipline](/en/academy/benchmark-discipline) and [Benchmark Scope](/en/validation/benchmark-scope) to make sure the reported number is even worth interpreting.
- Open the [Nsight Compute roofline-oriented documentation](/en/research/references#profiler-tooling-and-diagnosis-references) alongside [foundational performance-model reading](/en/research/references#papers-and-performance-mental-models).
- Compare the observed arithmetic-intensity story with the kernel stage you are reading: naive, tiled, double-buffered, or WMMA.

Questions to keep in mind:

- Did the optimization raise arithmetic intensity, reduce latency, or only move work around?
- Is the kernel limited by memory traffic, instruction mix, or launch geometry?
- Which evidence would justify saying the next optimization should target Tensor Cores instead of memory movement?

## Tensor Core constraints and fallback design

Study this route when WMMA looks fast in a chart but fragile in real workloads.

- Read [Tensor Core Path](/en/architecture/tensor-core-path) and [Tensor Core WMMA](/en/academy/kernel-tensor-core).
- Then open the [WMMA API reference and mixed-precision guidance](/en/research/references#official-cuda-and-nvidia-docs) to confirm fragment sizes, alignment requirements, and conversion costs.
- Finally, compare those constraints with the project's explicit fallback behavior so you can explain why unsupported shapes should degrade safely instead of silently producing misleading numbers.

Questions to keep in mind:

- Which input shapes are "Tensor Core friendly" and which are not?
- What part of the timing is actual matrix multiply work versus conversion or wrapper overhead?
- When is the FP32 fallback the more honest engineering choice?

## Profiling from symptoms to evidence

Study this route when you know a result changed but do not yet know why.

- Start with [Diagnosis Loop](/en/academy/diagnosis-loop).
- Use [Nsight Systems, Nsight Compute, and Compute Sanitizer references](/en/research/references#profiler-tooling-and-diagnosis-references) to separate launch issues, memory issues, and correctness issues.
- Cross-check with [Reproducibility](/en/validation/reproducibility) so your profiler session, benchmark run, and environment story stay aligned.

Questions to keep in mind:

- Is the symptom on the timeline, inside one kernel, or only in aggregate benchmark output?
- Which metric will tell you whether the bottleneck is bandwidth, occupancy, latency hiding, or invalid assumptions?
- What would you need to capture before making a public performance claim?

## Pick a route by current goal

| Goal | Best route |
|---|---|
| Build stronger intuition for shared-memory tiling | [GEMM tiling and hierarchy thinking](#gemm-tiling-and-hierarchy-thinking) |
| Learn to talk about occupancy without cargo-culting it | [Occupancy as a constraint, not a vanity metric](#occupancy-as-a-constraint-not-a-vanity-metric) |
| Explain performance limits with a better model | [Roofline thinking for SGEMM](#roofline-thinking-for-sgemm) |
| Understand when Tensor Cores help and when they complicate the story | [Tensor Core constraints and fallback design](#tensor-core-constraints-and-fallback-design) |
| Turn profiler output into a debugging plan | [Profiling from symptoms to evidence](#profiling-from-symptoms-to-evidence) |
