---
title: CUDA Memory Cheat Sheet
---

# CUDA Memory Cheat Sheet

This quick sheet is now part of the [Resources Hub](/en/resources/). Use it when you need to re-orient yourself before reopening kernel code, profiler output, or WMMA constraints.

## When this page is most useful

- Right before reading [Memory Flow](/en/architecture/memory-flow) or [Tiled Kernel](/en/kernel-tiled) again.
- When a benchmark changed and you need a fast checklist before blaming occupancy or Tensor Cores.
- When you want to explain memory behavior in an interview without reopening the full CUDA manuals.

## Coalescing quick rules

- Consecutive threads in a warp should touch consecutive addresses whenever possible.
- Accessing `B[k * N + col]` with large `N` can become stride-heavy for neighboring threads.
- Tiling is not only about reuse; it also reshapes access into more coalesced loads.

```mermaid
flowchart LR
    A[Global memory tile load] --> B[Shared memory tile]
    B --> C[Register accumulation]
    C --> D[Global memory writeback]
```

## Shared-memory watchpoints

| Question | Why it matters |
|---|---|
| Are threads writing a tile layout that later reads back contiguously? | Shared memory only helps when it fixes a global-memory access problem instead of creating a local one. |
| Did padding or index remapping remove bank conflicts on the hot path? | Bank conflicts can erase the benefit of otherwise-good tiling choices. |
| Did the extra shared-memory footprint change occupancy enough to matter? | Some tiling wins disappear if the launch geometry becomes too constrained. |

## Tensor Core memory notes

| Topic | What to remember |
|---|---|
| Alignment constraints | WMMA paths expect dimensions aligned to fragment-friendly sizes, typically 16, for efficient fragment handling. |
| Data conversion | End-to-end timing includes conversion and wrapper logic, not just the fused matrix multiply. |
| Safe behavior | Non-friendly shapes should fall back to the FP32 path instead of forcing a misleading WMMA result. |
| Reporting | Distinguish end-to-end numbers from compute-only numbers before comparing implementations. |

## Fast checklist when reading a kernel

1. Can I explain the global-memory access order for one warp?
2. Does shared-memory layout reduce conflicts rather than merely move data around?
3. Are register accumulators bounded and intentional?
4. Is Tensor Core fallback behavior explicit?
5. Do benchmark labels match what the kernel path really measures?

## Where to go next

- [Resources Hub](/en/resources/)
- [Further Reading Routes](/en/resources/further-reading)
- [Curated References](/en/references)
- [Diagnosis Loop](/en/methodology/diagnosis-loop)
