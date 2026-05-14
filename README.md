# SGEMM Optimization

[![CI](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/sgemm-optimization/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/sgemm-optimization/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)

English | [简体中文](README.zh-CN.md)

A CUDA SGEMM engineering notebook designed for both deep learning and interview presentation: from readable FP32 baselines to guarded Tensor Core WMMA, with cuBLAS-backed verification and explicit benchmark boundaries.

## Why this project stands out

- **Progressive kernel ladder**: naive -> tiled -> bank-conflict-free -> double-buffer -> Tensor Core.
- **Evidence-first reporting**: performance claims are paired with correctness policy and scope labels.
- **Comparable interfaces**: FP32 kernels share a unified `(A, B, C, M, K, N, stream)` launcher contract.
- **Interview-ready narrative**: dedicated pages for project highlights, interview walkthrough, and references.
- **Bilingual mirrored docs**: English and Chinese public pages stay aligned.

## Quick start

```bash
git clone https://github.com/LessUp/sgemm-optimization.git
cd sgemm-optimization

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/bin/sgemm_benchmark -a
ctest --test-dir build
```

Runtime tests and benchmarks require a CUDA-capable local machine. Hosted CI is limited to formatting, repository-structure, OpenSpec/governance, and Pages checks.

## Start here (GitHub Pages)

| Goal | Entry point |
|------|-------------|
| Open English home | [Docs Home](https://lessup.github.io/sgemm-optimization/en/) |
| Open Chinese home | [中文首页](https://lessup.github.io/sgemm-optimization/zh/) |
| Build and run once | [Getting Started](https://lessup.github.io/sgemm-optimization/en/getting-started) |
| Understand differentiation | [Project Highlights](https://lessup.github.io/sgemm-optimization/en/project-highlights) |
| Prepare interview explanation | [Interview Playbook](https://lessup.github.io/sgemm-optimization/en/interview-playbook) |
| Trace technical lineage | [References](https://lessup.github.io/sgemm-optimization/en/references) |
| Read normative specs | [OpenSpec Specs](openspec/specs/) |

## Validation boundary

| Environment | What to trust |
|-------------|---------------|
| Hosted CI | Formatting, docs/structure checks, OpenSpec governance, Pages buildability |
| Local CUDA GPU | Runtime correctness verification and benchmark performance |

This split is deliberate. CI keeps repository health; real GPU hardware validates runtime behavior and speed claims.

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
