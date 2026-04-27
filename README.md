# SGEMM Optimization

[![CI](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/sgemm-optimization/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/sgemm-optimization/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)

English | [简体中文](README.zh-CN.md)

Progressive CUDA SGEMM tutorial and reference implementation, from naive kernels to Tensor Core WMMA. Includes cuBLAS-backed verification, benchmark harness, and OpenSpec-governed repository rules.

## Why this repository exists

- **Show the optimization ladder clearly**: naive → tiled → bank-conflict-free → double-buffer → Tensor Core WMMA
- **Stay readable**: each optimization lives in its own kernel file with a consistent launch interface
- **Stay verifiable**: kernels are checked against cuBLAS with separate tolerances for FP32 and Tensor Core paths
- **Stay maintainable**: OpenSpec keeps docs, workflow, and validation rules aligned

## Optimization ladder

| Stage | Kernel | What you learn |
|------:|--------|----------------|
| 1 | [Naive](docs/kernel-naive/) | Thread-to-output mapping and baseline cost |
| 2 | [Tiled](docs/kernel-tiled/) | Shared-memory blocking and data reuse |
| 3 | [Bank-Free](docs/kernel-bank-free/) | Padding away 32-way bank conflicts |
| 4 | [Double Buffer](docs/kernel-double-buffer/) | Latency hiding through staged tiles |
| 5 | [Tensor Core](docs/kernel-tensor-core/) | WMMA usage with guarded FP32 fallback |

## Quick start

```bash
git clone https://github.com/LessUp/sgemm-optimization.git
cd sgemm-optimization

# Recommended: CMake
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/bin/sgemm_benchmark -a
ctest --test-dir build
```

```bash
# Quick local alternative
make GPU_ARCH=sm_86
make benchmark
make test
```

## Where to start

| If you want to... | Start here |
|-------------------|------------|
| Build and run once | [Getting Started](docs/getting-started/) |
| Learn the optimization path | [Learning Path](docs/learning-path/) |
| Understand repository structure | [Architecture](docs/architecture/) |
| See performance context | [Benchmark Results](docs/benchmark-results/) |
| Inspect governance rules | [Specifications](specs/) |

## Validation boundary

- **Local GPU machine**: runtime tests, correctness checks, benchmarking
- **GitHub Actions**: format/style, CUDA compile, OpenSpec checks, Pages deployment

Standard FP32 kernels: `rtol=1e-3`, `atol=1e-4`. Tensor Core path: `rtol=5e-2`, `atol=1e-2`.

## Repository layout

```text
src/
├── kernels/        # Five SGEMM kernel variants
├── utils/          # CUDA RAII, verification, benchmark helpers
└── main.cu         # Benchmark entry point
tests/
└── test_sgemm.cu   # Google Test suite
docs/               # Learning-oriented documentation
openspec/           # Stable specs, changes, workflow guidance
```

## Project status

This repository is in **archive-ready** state. All kernel implementations are complete, tests pass, and documentation is aligned. Non-trivial changes follow the OpenSpec workflow:

1. `/opsx:explore` — clarify scope and trade-offs
2. `/opsx:propose "description"` — create change artifacts
3. `/opsx:apply` — implement tasks
4. `/review` — quality gate
5. `/opsx:archive` — merge and close

Stable specs: `openspec/specs/`. Active changes: `openspec/changes/<change>/`.

## License

MIT. See [LICENSE](LICENSE).
