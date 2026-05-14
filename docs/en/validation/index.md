---
title: Validation
---

# Validation

This section explains **why the repository's performance claims are trustworthy, and where that trust stops**.

Methodology covers how we optimize. Validation covers the evidence boundary around those optimizations: correctness thresholds, benchmark scope labels, hosted-CI limits, and reproducibility expectations.

## The validation model

| Evidence surface | What it proves | What it does not prove |
|------------------|----------------|------------------------|
| Hosted CI + local structural checks | Docs/spec structure, Pages fitness, formatting/governance workflows, and repository health checks | GPU runtime correctness, CUDA benchmark numbers, or hardware-specific speedups |
| Local `ctest --test-dir build` on a GPU machine | Runtime correctness against the project's cuBLAS oracle | Universal performance claims |
| Local benchmark execution | Performance behavior on a named GPU, under a named command and scope label | Results on other GPUs, other CUDA stacks, or unlabeled workloads |

## Canonical validation pages

| Need | Page |
|------|------|
| Understand correctness thresholds and oracle policy | [Correctness Policy](/en/validation/correctness-policy) |
| Interpret benchmark labels and reported numbers | [Benchmark Scope](/en/validation/benchmark-scope) |
| Reproduce a result responsibly | [Reproducibility](/en/validation/reproducibility) |
| See a representative snapshot of results | [Benchmark Results](/en/benchmark-results) |

## What hosted CI proves

Hosted CI is trusted to prove repository health: documentation structure, Pages buildability, formatting checks, and OpenSpec/governance alignment. It keeps the public surface coherent.

Hosted CI is **not** trusted to prove CUDA runtime behavior or benchmark performance. Those claims require a real GPU machine.

## What only local GPU runs can prove

Local GPU runs are required for:

- cuBLAS-backed runtime correctness checks
- Tensor Core fast-path versus fallback behavior
- benchmark numbers, including end-to-end versus compute-only differences
- architecture-specific conclusions about occupancy, staging, and memory behavior

## How to read published numbers

Treat every number in this repository as **scoped evidence**, not a universal promise.

- Read the GPU model and CUDA context first.
- Read the benchmark label second.
- Read the shape set third.
- Only then compare the number to another result.

If any of those fields are missing, the number is a hint, not a claim.
