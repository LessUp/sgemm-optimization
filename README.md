# SGEMM Optimization

[![CI](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/sgemm-optimization/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/sgemm-optimization/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)

English | [简体中文](README.zh-CN.md)

Progressive CUDA SGEMM tutorial and reference implementation. The repository contains five hand-written kernel variants, cuBLAS-backed verification, a benchmark harness, and OpenSpec-governed repository rules for keeping the project compact and trustworthy.

## Why this repository exists

- **Show the optimization ladder clearly**: naive -> tiled -> bank-conflict-free -> double-buffered -> Tensor Core WMMA
- **Stay readable**: each optimization lives in its own kernel file and keeps a consistent launch interface
- **Stay verifiable**: kernels are checked against cuBLAS, with separate tolerances for FP32 and Tensor Core paths
- **Stay maintainable**: the repository uses OpenSpec to keep docs, workflow, and validation rules aligned

## Kernel progression

| Stage | File | Main idea |
|-------|------|-----------|
| Naive | `src/kernels/naive_sgemm.cuh` | Baseline triple-loop mapping |
| Tiled | `src/kernels/tiled_sgemm.cuh` | Shared-memory blocking |
| Bank-Free | `src/kernels/bank_conflict_free_sgemm.cuh` | `[TILE_SIZE][TILE_SIZE+1]` padding |
| Double Buffer | `src/kernels/double_buffer_sgemm.cuh` | Tile staging overlap and latency hiding |
| Tensor Core | `src/kernels/tensor_core_sgemm.cuh` | WMMA path with safe FP32 fallback |

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

## Validation model

- **Local GPU machine**: runtime tests, correctness checks, and benchmarking
- **GitHub Actions**: format/style, CUDA compile validation, OpenSpec/repository checks, and Pages deployment

Standard FP32 kernels use `rtol=1e-3`, `atol=1e-4`. The Tensor Core path uses `rtol=5e-2`, `atol=1e-2`.

## Read next

- [Getting Started](docs/getting-started.md)
- [Learning Path](docs/learning-path.md)
- [Architecture Overview](docs/architecture.md)
- [Benchmark Notes](docs/benchmark-results.md)
- [Specifications Index](specs.md)
- [GitHub Pages site](https://lessup.github.io/sgemm-optimization/)

## Repository layout

```text
src/
├── kernels/        # Five SGEMM kernel variants
├── utils/          # CUDA RAII, verification, benchmark helpers
└── main.cu         # Benchmark entry point
tests/
└── test_sgemm.cu   # Google Test suite
docs/               # Public learning-oriented documentation
openspec/           # Stable specs, changes, and workflow guidance
```

## Development workflow

Non-trivial repository changes are expected to follow:

1. `/opsx:explore`
2. `/opsx:propose "description"`
3. `/opsx:apply`
4. `/review`
5. `/opsx:archive`

The stable authoritative specs live under `openspec/specs/`. Active implementation plans live under `openspec/changes/<change>/`.

## License

MIT. See [LICENSE.md](LICENSE.md).
