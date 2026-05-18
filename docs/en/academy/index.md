---
title: Academy
---

# Academy

The academy is the ordered learning surface of this repository. Architecture gives the system map. The academy gives the teaching sequence — the order in which each kernel stage is explained, and why that order is non-negotiable.

## The structuring principle

Read kernels as a progression of bottleneck shifts, not as a list of tricks:

| Stage | Bottleneck exposed | Structural change introduced |
|---|---|---|
| Naïve FP32 | Unlimited DRAM traffic | Establishes the cost model |
| Tiled FP32 | Redundant global reads | Shared-memory staging |
| Bank-Free FP32 | Shared-memory bank conflicts | Tile padding |
| Double Buffer | Memory latency in critical path | Overlap staging and compute |
| Tensor Core WMMA | FP32 throughput ceiling | Hardware fragment accumulation |

Each later page assumes the previous page already explained why its extra complexity is justified. Reading out of order makes the causal chain invisible.

## Academy map

| Track | Purpose | Start here |
|---|---|---|
| Orientation | Learn the route through the ladder before opening any kernel page | [Learning Path](./learning-path) |
| Experiment discipline | Avoid drawing conclusions from sloppy measurements | [Benchmark Discipline](./benchmark-discipline) |
| Bottleneck reasoning | Turn symptoms into the next defendable architectural change | [Diagnosis Loop](./diagnosis-loop) |
| Kernel deep dives | Inspect the actual optimization stages in sequence | [Naive Kernel](./kernel-naive) |
| Retention aids | Refresh memory hierarchy and tuning heuristics quickly | [CUDA Memory Cheat Sheet](./cuda-memory-cheatsheet) |

## Recommended reading order

1. [Learning Path](./learning-path) — orientation before any kernel
2. [Naive Kernel](./kernel-naive) — cost model baseline
3. [Tiled Kernel](./kernel-tiled) — shared-memory reuse
4. [Bank Conflict Free](./kernel-bank-free) — stability under conflict shapes
5. [Double Buffer](./kernel-double-buffer) — latency hiding
6. [Tensor Core WMMA](./kernel-tensor-core) — guarded throughput ceiling
7. [Diagnosis Loop](./diagnosis-loop) — turn measurements into decisions
8. [Optimization Playbook](./optimization-playbook) — structured tuning process

## Interview-ready framing

When defending any kernel stage under review, use this four-part structure:

1. **Name the current bottleneck** — what resource is saturated or wastefully used?
2. **Name the specific structural change** — what does this kernel do differently at the hardware level?
3. **State the evidence requirement** — what measurement would confirm the change helped?
4. **State the constraint** — what assumption or shape condition limits this improvement?

That sequence keeps the discussion at the level of engineering reasoning rather than benchmark screenshots. The academy is designed to give you a defensible answer for each of the five stages.

## What the academy is not

The academy is not a reference manual for CUDA programming. For reference, use the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) and the [CUDA Memory Cheat Sheet](./cuda-memory-cheatsheet) in this section.

The academy is not a substitute for reading the source code. Each kernel page explains the architectural reasoning; the code itself contains the implementation. Both are necessary to give a complete account of any stage.

## Related resources

- [Architecture Overview](../architecture/) — the system map that contextualizes the ladder
- [Validation Overview](../validation/) — the trust boundary for any number produced during academy study
- [Performance Model](../validation/performance-model) — analytical cost model behind each ladder stage
