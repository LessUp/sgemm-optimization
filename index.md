---
layout: default
title: SGEMM Optimization
---

# SGEMM Optimization: From Naive to Tensor Core

[![CI](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/sgemm-optimization/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/sgemm-optimization/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/LessUp/sgemm-optimization/blob/main/LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
[![GitHub stars](https://img.shields.io/github/stars/LessUp/sgemm-optimization?style=social)](https://github.com/LessUp/sgemm-optimization/stargazers)

**Hand-written, progressively optimized CUDA matrix multiplication — the "Hello World" of HPC**

Five progressively optimized kernel variants demonstrating core GPU optimization techniques, from a naive triple-loop to Tensor Core WMMA API.

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/LessUp/sgemm-optimization.git
cd sgemm-optimization

# Build (CMake recommended)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Benchmark
./build/bin/sgemm_benchmark -a
```

[📖 Getting Started Guide](docs/getting-started.md) · [🏗️ Architecture](docs/architecture.md) · [📊 Benchmarks](docs/benchmark-results.md)

---

## 📊 Performance Summary

NVIDIA RTX 3060 Laptop (Ampere, sm_86), 1024×1024×1024:

| Kernel | GFLOPS | vs cuBLAS | Time | Key Technique |
|--------|-------:|----------:|------:|---------------|
| **cuBLAS** | 5727 | 100% | 0.375 ms | NVIDIA library |
| **Tensor Core** | 2300 | 40.2% | 0.934 ms | WMMA API, FP16 |
| **Tiled** | 753 | 13.1% | 2.853 ms | Shared memory |
| **Double Buffer** | 701 | 12.2% | 3.064 ms | Pipeline overlap |
| **Bank-Free** | 673 | 11.8% | 3.190 ms | Padding |
| **Naive** | 604 | 10.6% | 3.553 ms | Baseline |

> Performance varies by GPU, CUDA version, and matrix size. [See full benchmarks](docs/benchmark-results.md).

---

## 🔄 Optimization Journey

```
┌─────────┐    ┌──────────┐    ┌──────────────┐    ┌───────────────┐
│  Naive  │───▶│  Tiled   │───▶│  Bank-Free   │───▶│ Double Buffer │
└─────────┘    └──────────┘    └──────────────┘    └───────┬───────┘
                                                         │
                                                         ▼
                                             ┌───────────────────┐
                                             │   Tensor Core     │
                                             │   (WMMA API)      │
                                             └───────────────────┘
```

| Stage | What Changes | Why It Helps |
|-------|-------------|--------------|
| **Naive → Tiled** | Shared memory tiles | Data reuse, less global memory traffic |
| **Tiled → Bank-Free** | `[32][33]` padding | Eliminates 32-way bank conflicts |
| **Bank-Free → Double Buffer** | Two buffers, async load | Hides memory latency |
| **→ Tensor Core** | WMMA `mma_sync` | Dedicated matrix units, ~8× peak |

---

## 🛠️ Core Techniques

### Memory Coalescing
Consecutive threads access consecutive addresses → bandwidth utilization from ~12.5% to ~100%.

### Shared Memory Tiling
Load tiles into shared memory → global memory accesses reduced by `TILE_SIZE×`.

### Bank Conflict Elimination
Padding `[32][33]` instead of `[32][32]` → shared memory bandwidth restored.

### Double Buffering
Two shared-memory buffers alternate load/compute → memory latency hidden.

### Tensor Core (WMMA)
FP16 input + FP32 accumulation → dedicated hardware matrix units.

---

## 📁 Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | Build, run, and benchmark in 5 minutes |
| [Architecture Guide](docs/architecture.md) | System design and interface specifications |
| [Kernel Details](docs/kernel-details.md) | Deep dive into each kernel implementation |
| [Benchmark Results](docs/benchmark-results.md) | Performance analysis and roofline model |
| [CHANGELOG](CHANGELOG.md) | Version history |
| [Contributing](CONTRIBUTING.md) | How to contribute |

### Specifications (SDD)

This project follows **Spec-Driven Development**:

- [Product Requirements](specs/product/sgemm-kernel-requirements.md)
- [Core Architecture RFC](specs/rfc/0001-core-architecture.md)
- [Implementation Roadmap](specs/rfc/0002-implementation-roadmap.md)
- [Test Specifications](specs/testing/kernel-verification.md)

---

## 🧪 Testing

Google Test coverage includes:

| Property | What It Verifies |
|----------|-----------------|
| **Numerical Correctness** | Standard kernels match cuBLAS |
| **Tensor Core Fast Path** | WMMA validated on 16-aligned sizes |
| **Tensor Core Fallback** | Non-aligned sizes safely fall back to FP32 |
| **Edge Cases** | 1×1×1 and unaligned dimensions |

```bash
# Run tests
cmake --build build --target test_sgemm
ctest --test-dir build
```

---

## 📖 GPU Architecture Support

| GPU | Architecture | Flag |
|-----|-------------|------|
| Tesla V100 | Volta | `sm_70` |
| RTX 3090 / A100 | Ampere | `sm_86` |
| RTX 4090 | Ada Lovelace | `sm_89` |
| H100 | Hopper | `sm_90` |

---

## 📚 References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA's high-performance GEMM library
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Roofline Model](https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/)

---

<div style="text-align: center; margin-top: 2rem;">
  <a href="https://github.com/LessUp/sgemm-optimization">🔍 View on GitHub</a> ·
  <a href="README.md">English README</a> ·
  <a href="README.zh-CN.md">中文 README</a> ·
  <a href="CHANGELOG.md">📜 Changelog</a>
</div>
