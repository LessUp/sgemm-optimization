---
title: Validation
---

# Validation

This section explains **why the repository's performance claims are trustworthy, and where that trust stops**.

Optimization methodology answers: "how do we improve performance?" Validation answers a different question: "what does the evidence actually prove, and what can it not prove?" Both questions matter. Only the second is honest.

## The validation model

| Evidence surface | What it proves | What it does not prove |
|---|---|---|
| Hosted CI + repository checks | Formatting, CUDA compilation, docs structure, Pages fitness, and route/workflow health checks | GPU runtime correctness, CUDA benchmark numbers, or hardware-specific speedups |
| Local `ctest --test-dir build` on a GPU machine | Runtime correctness against the project's cuBLAS oracle, under the project's numerical tolerance policy | Universal performance claims or GPU model independence |
| Local benchmark execution | Performance behavior on a named GPU, under a named command and scope label | Results on other GPUs, other CUDA stacks, or unlabeled workloads |

## Canonical validation pages

| Need | Page |
|---|---|
| Understand correctness thresholds and oracle policy | [Correctness Policy](./correctness-policy) |
| Interpret benchmark labels and reported numbers | [Benchmark Scope](./benchmark-scope) |
| Reproduce a result responsibly | [Reproducibility](./reproducibility) |
| See a representative snapshot of results | [Benchmark Results](./benchmark-results) |
| Understand the analytical model behind the speedups | [Performance Model](./performance-model) |

## What hosted CI proves

Hosted CI is trusted to prove repository health: CUDA compilation, documentation structure, Pages buildability, formatting checks, and docs test/build health. It keeps the public surface coherent.

Hosted CI is **not** trusted to prove CUDA runtime behavior or benchmark performance. Those claims require a real GPU machine.

## What only local GPU runs can prove

Local GPU runs are required for:

- cuBLAS-backed runtime correctness checks
- Tensor Core fast-path versus fallback behavior
- benchmark numbers, including end-to-end versus compute-only differences
- architecture-specific conclusions about occupancy, staging, and memory behavior

## How to read published numbers

Treat every number in this repository as **scoped evidence**, not a universal promise.

1. Read the GPU model and CUDA context first.
2. Read the benchmark label second.
3. Read the shape set third.
4. Only then compare the number to another result.

If any of those fields are missing, the number is a hint, not a claim.

## Common presentation mistakes this project avoids

- Claiming "Tensor Core is always faster" without shape, conversion, and fallback caveats.
- Quoting one GFLOPS number without its benchmark label or workload scope.
- Ignoring the numerical-tolerance difference between FP32 and mixed precision.
- Treating hosted CI success as proof of CUDA runtime correctness or performance.
- Comparing results across different GPU models or CUDA versions without explicit labeling.

## The trust hierarchy

```
Most trustworthy
  │
  ▼  [Local GPU] cuBLAS oracle + ctest correctness suite
  ▼  [Local GPU] Benchmark with named hardware, shape, and label
  ▼  [Hosted CI] Formatting, CUDA compilation, structure, docs, and workflow checks
  ▼  [Analytical] Performance model (roofline, cost model)
  ▼  [Qualitative] Architecture rationale and design arguments
  │
Least falsifiable
```

The validation section operates at the top two levels. The architecture and academy sections operate at the bottom two. The separation is intentional.
