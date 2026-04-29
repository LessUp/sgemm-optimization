# SGEMM Optimization

[![CI](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/sgemm-optimization/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/sgemm-optimization/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)

English | [简体中文](README.zh-CN.md)

A compact CUDA SGEMM learning project that walks from a readable baseline kernel to Tensor Core WMMA, with cuBLAS verification and a CMake-first build.

## What makes it useful

- **One optimization ladder**: naive -> tiled -> bank-conflict-free -> double-buffer -> Tensor Core.
- **Comparable kernel interfaces**: every FP32 kernel uses the same `(A, B, C, M, K, N, stream)` launcher shape.
- **Verification-first harness**: kernel output is checked against cuBLAS with separate tolerances for FP32 and Tensor Core paths.
- **Learning-oriented docs**: GitHub Pages carries the full walkthrough instead of duplicating it in the README.

## Quick start

```bash
git clone https://github.com/LessUp/sgemm-optimization.git
cd sgemm-optimization

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/bin/sgemm_benchmark -a
ctest --test-dir build
```

Runtime tests and benchmarks require a CUDA-capable local machine. Hosted CI is limited to compile-time, formatting, repository-structure, OpenSpec, and Pages checks.

## Start here

| Goal | Entry point |
|------|-------------|
| Use the project site | [GitHub Pages](https://lessup.github.io/sgemm-optimization/) |
| Build and run once | [Getting Started](docs/getting-started.md) |
| Follow the kernel ladder | [Learning Path](docs/learning-path.md) |
| Inspect the source layout | [Architecture](docs/architecture.md) |
| Read the normative specs | [Specifications](specs.md) |

## Source map

```text
src/kernels/   CUDA SGEMM implementations
src/utils/     CUDA RAII, verification, benchmark helpers
src/main.cu    benchmark CLI
tests/         Google Test coverage against cuBLAS
docs/          learning documentation mirrored on Pages
openspec/      stable specs and change workflow
```

## License

MIT. See [LICENSE.md](LICENSE.md).
