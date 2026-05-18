---
title: Academy
---

# Academy

The academy is the ordered learning surface of this repository. Architecture gives the map. The academy gives the teaching sequence.

## The rule of this section

Read kernels as a progression of bottleneck shifts:

1. establish the cost model
2. change data reuse
3. stabilize shared-memory behavior
4. overlap staging and compute
5. introduce guarded mixed precision

That order matters because every later page assumes the previous page already explained why its extra complexity is justified.

## Academy map

| Track | Purpose | Start here |
|---|---|---|
| Orientation | Learn the route through the ladder | [Learning Path](./learning-path) |
| Experiment discipline | Avoid drawing conclusions from sloppy measurements | [Benchmark Discipline](./benchmark-discipline) |
| Bottleneck reasoning | Turn symptoms into the next defendable change | [Diagnosis Loop](./diagnosis-loop) |
| Kernel deep dives | Inspect the actual optimization stages | [Naive Kernel](./kernel-naive) |
| Retention aids | Refresh memory and tuning heuristics quickly | [CUDA Memory Cheat Sheet](./cuda-memory-cheatsheet) |

## Recommended order

1. [Learning Path](./learning-path)
2. [Naive Kernel](./kernel-naive)
3. [Tiled Kernel](./kernel-tiled)
4. [Bank Conflict Free](./kernel-bank-free)
5. [Double Buffer](./kernel-double-buffer)
6. [Tensor Core WMMA](./kernel-tensor-core)
7. [Diagnosis Loop](./diagnosis-loop)
8. [Optimization Playbook](./optimization-playbook)

## Interview-ready framing

When you need to explain the project quickly:

1. name the current bottleneck
2. name the specific structural change
3. say what evidence would prove that change helped
4. say what constraint still limits the design

That sequence keeps the discussion technical and keeps you out of vague “it got faster” claims.
