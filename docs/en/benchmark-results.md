---
title: Benchmark Results
---

# Benchmark Results

Representative performance notes, not a universal promise



## Reference snapshot

Sample numbers from an RTX 3060 Laptop at `1024 x 1024 x 1024`:

| Kernel | GFLOPS | vs cuBLAS |
|--------|-------:|----------:|
| cuBLAS | 5727 | 100.0% |
| Tensor Core (WMMA compute-only) | 2300 | 40.2% |
| Tiled | 753 | 13.1% |
| Double Buffer | 701 | 12.2% |
| Bank-Free | 673 | 11.8% |
| Naive | 604 | 10.6% |



## Tensor Core note

The benchmark reports:

- **WMMA end-to-end**: the safe FP32 wrapper, including conversion and fallback handling
- **WMMA compute-only**: the pure pre-converted FP16 path, shown only when `M`, `K`, and `N` are multiples of 16

When the dimensions are not Tensor Core friendly, the implementation falls back to a safer FP32 path instead of forcing WMMA.



## Related references

- [Getting Started](/en/getting-started)
- [Learning Path](/en/learning-path)
- [Kernel progression in README](https://github.com/LessUp/sgemm-optimization/blob/master/README.md)
