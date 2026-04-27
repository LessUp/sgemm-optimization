---
layout: default
title: Benchmark Results
nav_order: 9
permalink: /docs/benchmark-results/
lang: en
page_key: benchmark-results
lang_ref: zh-benchmark-results
---

# Benchmark Results
{: .fs-8 }

Representative performance notes, not a universal promise
{: .fs-6 .fw-300 }

---

## Read this first

Performance depends on GPU model, CUDA version, clocks, thermals, and matrix shape. Treat the numbers below as a **reference snapshot** that helps explain the optimization ladder, not as a guarantee for every machine.

---

## Reference snapshot

Sample numbers from an RTX 3060 Laptop at `1024 x 1024 x 1024`:

| Kernel | GFLOPS | vs cuBLAS |
|--------|-------:|----------:|
| cuBLAS | 5727 | 100.0% |
| Tensor Core | 2300 | 40.2% |
| Tiled | 753 | 13.1% |
| Double Buffer | 701 | 12.2% |
| Bank-Free | 673 | 11.8% |
| Naive | 604 | 10.6% |

---

## What matters more than the exact number

| Transition | Main lesson |
|------------|-------------|
| Naive -> Tiled | Shared-memory reuse matters immediately |
| Tiled -> Bank-Free | Memory layout details can remove hidden bottlenecks |
| Bank-Free -> Double Buffer | Overlap and staging help when memory stalls dominate |
| Double Buffer -> Tensor Core | Specialized hardware changes the ceiling dramatically |

---

## Tensor Core note

The benchmark reports:

- **WMMA end-to-end**: includes conversion and fallback handling
- **WMMA compute-only**: shown only when `M`, `K`, and `N` are multiples of 16

When the dimensions are not Tensor Core friendly, the implementation falls back to a safer FP32 path instead of forcing WMMA.

---

## Reproduce on your machine

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

./build/bin/sgemm_benchmark -a
./build/bin/sgemm_benchmark --dims 256 384 640
```

If you want longer measurements:

```bash
./build/bin/sgemm_benchmark -a --warmup 10 --benchmark 50
```

---

## Related references

- [Getting Started](getting-started/)
- [Learning Path](learning-path/)
- [Kernel progression in README](https://github.com/LessUp/sgemm-optimization/blob/master/README.md)
