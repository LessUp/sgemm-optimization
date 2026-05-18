---
title: Benchmark Results
---

# Benchmark Results

Representative performance notes, not a universal promise.

This page is now only the **results snapshot** surface. For trust policy and interpretation rules, use [Validation](/en/validation/). For experiment design, use [Methodology](/en/academy/).

## Reference snapshot

### End-to-end snapshot

Sample numbers from an RTX 3060 Laptop at `1024 x 1024 x 1024`:

| Kernel | GFLOPS | vs cuBLAS |
|--------|-------:|----------:|
| cuBLAS | 5727 | 100.0% |
| Tiled | 753 | 13.1% |
| Double Buffer | 701 | 12.2% |
| Bank-Free | 673 | 11.8% |
| Naive | 604 | 10.6% |

### Compute-only WMMA snapshot

The repository also reports a narrower fast-path measurement for Tensor Core-friendly shapes:

| Kernel | GFLOPS | vs cuBLAS |
|--------|-------:|----------:|
| Tensor Core (WMMA compute-only) | 2300 | 40.2% |

The benchmark harness also emits **WMMA end-to-end** results, but this condensed page does not publish a single headline number for that path because FP32→FP16 conversion and fallback behavior depend strongly on the exact local execution path. Treat the compute-only row as an upper-bound reference, then use [Benchmark Scope](/en/validation/benchmark-scope) and local runs to interpret the full end-to-end gap.

## How to read this page

- These numbers are a **representative local snapshot**, not a promise for every GPU.
- Compare them only after reading [Benchmark Scope](/en/validation/benchmark-scope).
- Read the end-to-end table first; treat `WMMA compute-only` as a narrow fast-path label, not a replacement for end-to-end behavior.
- Assume hosted CI did **not** prove these numbers. Only local GPU runs can.

## Tensor Core note

The benchmark suite reports:

- **WMMA end-to-end**: the safe FP32 wrapper, including conversion and fallback handling
- **WMMA compute-only**: the pure pre-converted FP16 path, shown only when `M`, `K`, and `N` are multiples of 16

When dimensions are not Tensor Core friendly, the implementation falls back to a safer FP32 path instead of forcing WMMA.

## Read together with

- [Validation Overview](/en/validation/)
- [Correctness Policy](/en/validation/correctness-policy)
- [Benchmark Scope](/en/validation/benchmark-scope)
- [Reproducibility](/en/validation/reproducibility)
- [Methodology](/en/academy/)
