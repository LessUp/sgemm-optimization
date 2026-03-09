---
layout: default
title: SGEMM Optimization
---

# SGEMM Optimization: From Naive to Tensor Core

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)

Hand-written, progressively optimized CUDA matrix multiplication — the "Hello World" of HPC. Five kernel variants demonstrate core GPU optimization techniques, from a naive triple loop to **Tensor Core WMMA reaching 40% of cuBLAS throughput**.

## Benchmark Results (RTX 3060 Laptop, 1024×1024)

| Kernel | GFLOPS | vs cuBLAS | Key Technique |
|--------|--------|-----------|---------------|
| **cuBLAS** (reference) | 5727 | 100% | — |
| **Tensor Core** (WMMA) | 2300 | 40.2% | FP16 → FP32 mixed precision |
| **Tiled** (32×32) | 753 | 13.1% | Shared memory blocking |
| **Double Buffer** | 701 | 12.2% | Compute-memory overlap |
| **Bank Conflict Free** | 673 | 11.8% | Shared memory padding (+1) |
| **Naive** | 604 | 10.6% | One thread per output element |

*All kernels verified against cuBLAS (allclose: rtol=1e-3, atol=1e-4; Tensor Core: rtol=5e-2)*

## Optimization Roadmap

```
 Naive (604)  ──▶  Tiled (753)  ──▶  Bank-Free (673)  ──▶  Double Buffer (701)
                                                                     │
                                              Tensor Core (2300)  ◀──┘
```

| Stage | What Changes | Why It Helps |
|-------|-------------|--------------|
| **Naive → Tiled** | Load tiles into shared memory | Data reuse reduces global memory traffic by TILE_SIZE× |
| **Tiled → Bank-Free** | Pad shared memory `[32][33]` | Eliminates 32-way bank conflicts on column access |
| **Bank-Free → Double Buffer** | Two shared-memory buffers | Overlaps next-tile load with current-tile compute |
| **→ Tensor Core** | WMMA API `mma_sync` | Dedicated matrix units, ~8× peak over CUDA cores |

## Key Optimization Details

**Memory Coalescing** — Naive accesses matrix B columns non-contiguously (stride = N). Tiled loading ensures warp-wide coalesced reads (stride = 1), improving bandwidth utilization from ~12.5% to near 100%.

**Shared Memory Tiling** — Each tile is loaded once from global memory and reused TILE_SIZE times in shared memory. Global memory traffic drops from O(N³) to O(N³/TILE_SIZE).

**Tensor Core (WMMA)** — A single warp cooperatively executes a 16×16×16 matrix multiply using `nvcuda::wmma` fragments. FP16 inputs with FP32 accumulation provide ~8× peak FLOPS over standard CUDA cores on Ampere.

**Roofline Analysis** — SGEMM arithmetic intensity ≈ N/3 for square matrices. Small matrices (N < 256) are memory-bound; large matrices (N > 1024) are compute-bound.

## Quick Start

```bash
# Build (adjust GPU arch for your hardware)
make GPU_ARCH=sm_86

# Run benchmark suite
make benchmark
```

### Sample Output

```
Kernel              | Dimensions         |    Time |  Performance | Pass
-----------------------------------------------------------------------
cuBLAS              | 1024 x 1024 x 1024 | 0.375ms | 5726 GFLOPS  | PASS
Naive               | 1024 x 1024 x 1024 | 3.553ms |  604 GFLOPS  | PASS
Tiled (32x32)       | 1024 x 1024 x 1024 | 2.853ms |  753 GFLOPS  | PASS
Bank Conflict Free  | 1024 x 1024 x 1024 | 3.190ms |  673 GFLOPS  | PASS
Double Buffer       | 1024 x 1024 x 1024 | 3.064ms |  701 GFLOPS  | PASS
Tensor Core (WMMA)  | 1024 x 1024 x 1024 | 0.934ms | 2300 GFLOPS  | PASS
```

## Testing

Property-based tests with Google Test:

| Property | What It Verifies |
|----------|-----------------|
| Numerical correctness | All kernels match cuBLAS output |
| Tensor Core tolerance | Correct under relaxed FP16 tolerance |
| Error detection | Verification system catches injected errors |
| Dimension invariance | All kernels handle arbitrary aligned sizes |

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | CUDA C++17 |
| Build | Makefile |
| Dependencies | cuBLAS, Google Test (optional) |
| GPU | Compute Capability 7.0+ (Volta → Hopper) |

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) — Simon Boehm
- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA's high-performance GEMM library
- [Roofline Model](https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/)

---

[View on GitHub](https://github.com/LessUp/sgemm-optimization) · [README](README.md)
