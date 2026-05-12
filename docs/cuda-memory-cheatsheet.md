---
layout: default
title: CUDA Memory Cheat Sheet
parent: Home
nav_order: 11
permalink: /docs/cuda-memory-cheatsheet/
lang: en
page_key: cuda-memory-cheatsheet
lang_ref: zh-cuda-memory-cheatsheet
---

# CUDA Memory Cheat Sheet
{: .fs-8 }

A compact memory reference for reading SGEMM kernels faster
{: .fs-6 .fw-300 }

---

## Memory hierarchy in one map

| Layer | Typical scope | SGEMM use in this repository | Common pitfall |
|------|----------------|------------------------------|----------------|
| Global memory | Device-wide | Input/output matrices and staging buffers | Uncoalesced access pattern |
| Shared memory | Per block | Tile reuse for A/B submatrices | Bank conflicts from layout |
| Register | Per thread | Accumulators and fragment values | Register pressure lowers occupancy |
| L2 cache | Device-wide cache | Helps repeated global reads | Assuming cache always hides bad indexing |

---

## Coalescing quick rules

- Consecutive threads in a warp should touch consecutive addresses whenever possible.
- Accessing `B[k * N + col]` with large `N` can become stride-heavy for neighboring threads.
- Tiling is not just reuse; it is also a way to reshape access into coalesced loads.

```mermaid
flowchart LR
    A[Global memory tile load] --> B[Shared memory tile]
    B --> C[Register accumulation]
    C --> D[Global memory writeback]
```

---

## Shared-memory bank conflict cues

| Pattern | Risk level | Typical fix |
|---------|------------|-------------|
| `tile[ty][tx]` with aligned warp access | Low | Keep contiguous thread-to-column mapping |
| `tile[tx][ty]` transpose-like without padding | High | Add padding like `[32][33]` |
| Multiple stages reuse same address stride | Medium | Revisit stage layout and lane mapping |

Bank conflicts do not always dominate, but they are easy to miss and easy to overestimate.

---

## Tensor Core memory notes

| Topic | What to remember |
|-------|------------------|
| Alignment constraints | WMMA path expects dimensions aligned to 16 for efficient fragment handling |
| Data conversion | End-to-end timing includes conversion and wrapper logic |
| Safe behavior | Non-friendly shapes should fall back to FP32 path |
| Reporting | Distinguish end-to-end and compute-only results |

---

## Profiler-to-action map

| Metric trend | Interpretation | Next action |
|--------------|----------------|-------------|
| High `dram` throughput, low `sm` throughput | Memory is likely limiting | Improve coalescing and reuse |
| Low occupancy with high register usage | Registers throttle active warps | Reduce per-thread temporary state |
| Shared bank conflict spikes | Shared layout mismatch | Apply padding or remap lanes |

---

## Fast checklist when reading a kernel

1. Can I explain the global memory access order for one warp?
2. Is shared memory layout conflict-aware?
3. Are register accumulators bounded and intentional?
4. Is Tensor Core fallback behavior explicit?
5. Do benchmark labels match what the kernel path really measures?

---

## Related pages

- [Kernel 1: Naive](kernel-naive/)
- [Kernel 3: Bank Conflict Free](kernel-bank-free/)
- [Kernel 5: Tensor Core](kernel-tensor-core/)
- [Optimization Playbook](optimization-playbook/)
