---
title: CUDA Memory Cheat Sheet
---

# CUDA Memory Cheat Sheet

A compact memory reference for reading SGEMM kernels faster



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



## Tensor Core memory notes

| Topic | What to remember |
|-------|------------------|
| Alignment constraints | WMMA path expects dimensions aligned to 16 for efficient fragment handling |
| Data conversion | End-to-end timing includes conversion and wrapper logic |
| Safe behavior | Non-friendly shapes should fall back to FP32 path |
| Reporting | Distinguish end-to-end and compute-only results |



## Fast checklist when reading a kernel

1. Can I explain the global memory access order for one warp?
2. Is shared memory layout conflict-aware?
3. Are register accumulators bounded and intentional?
4. Is Tensor Core fallback behavior explicit?
5. Do benchmark labels match what the kernel path really measures?

---
