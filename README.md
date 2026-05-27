# SGEMM Optimization

[![CI](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/sgemm-optimization/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/sgemm-optimization/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)

English | [简体中文](README.zh-CN.md)

This repository is a CUDA SGEMM case study presented as a technical whitepaper and kernel academy. It starts from readable FP32 baselines, climbs through tiled, bank-conflict-aware, double-buffer, and guarded Tensor Core WMMA paths, then frames every performance claim with explicit validation boundaries.

## Why it stands out

- **Readable optimization ladder**: every kernel stage exists to expose one bottleneck shift.
- **Evidence-first public story**: correctness policy, benchmark scope, and local-versus-CI trust boundaries stay attached to every claim.
- **Interview-grade positioning**: the Pages site is written so the project can be explained, defended, and audited under technical pressure.
- **Bilingual mirrored docs**: English and Chinese routes stay structurally aligned across the full public site.

## Quick start

```bash
git clone https://github.com/LessUp/sgemm-optimization.git
cd sgemm-optimization

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/bin/sgemm_benchmark -a
ctest --test-dir build
```

Runtime tests and benchmarks require a local CUDA-capable machine. Hosted CI validates formatting, CUDA compilation, OpenSpec/governance structure, and Pages buildability.

## GitHub Pages entry points

The README is the executive summary. The long-form technical narrative lives on Pages.

| Goal | Entry point |
|------|-------------|
| Open English home | [English Home](https://lessup.github.io/sgemm-optimization/en/) |
| Open Chinese home | [中文首页](https://lessup.github.io/sgemm-optimization/zh/) |
| Get oriented quickly | [Project Guide](https://lessup.github.io/sgemm-optimization/en/overview/) |
| Inspect system structure | [Architecture](https://lessup.github.io/sgemm-optimization/en/architecture/) |
| Study the kernel ladder | [Academy](https://lessup.github.io/sgemm-optimization/en/academy/) |
| Check what the evidence proves | [Validation](https://lessup.github.io/sgemm-optimization/en/validation/) |
| Trace papers and related repos | [Research Desk](https://lessup.github.io/sgemm-optimization/en/research/) |
| Read normative repository requirements | [OpenSpec Specs](openspec/specs/) |

## Validation boundary

| Environment | What it can prove |
|-------------|-------------------|
| Hosted CI | Formatting, CUDA compilation, OpenSpec/governance structure, Pages buildability |
| Local CUDA GPU | Runtime correctness, fallback behavior, benchmark performance |

This split is deliberate. CI catches build and structure issues early, but only local GPU execution can validate runtime behavior and speed claims.

## Source map

```text
src/kernels/   CUDA SGEMM implementations
src/utils/     CUDA RAII, verification, benchmark helpers
src/main.cu    benchmark CLI
tests/         Google Test coverage against cuBLAS
docs/          VitePress whitepaper and academy, mirrored under /en and /zh
openspec/      stable specs and change workflow
```

## License

MIT. See [LICENSE.md](LICENSE.md).
