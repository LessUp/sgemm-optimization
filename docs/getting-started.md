---
layout: default
title: Getting Started
parent: Home
nav_order: 1
permalink: /docs/getting-started/
lang: en
page_key: getting-started
lang_ref: zh-getting-started
---

# Getting Started
{: .fs-8 }

Build, run, and validate the project without guessing the toolchain
{: .fs-6 .fw-300 }

---

## Requirements

| Item | Requirement |
|------|-------------|
| GPU | NVIDIA Volta (`sm_70`) or newer |
| CUDA Toolkit | 11.0+ |
| CMake | 3.18+ |
| Host compiler | GCC 9+ or Clang 10+ |

Tensor Core benchmarks require `sm_70+` and dimensions aligned to 16. The code still runs on the guarded FP32 path when those conditions are not met.

---

## Recommended build flow

```bash
git clone https://github.com/LessUp/sgemm-optimization.git
cd sgemm-optimization

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Run the default benchmark:

```bash
./build/bin/sgemm_benchmark
```

Run the broader benchmark set:

```bash
./build/bin/sgemm_benchmark -a
```

Run tests:

```bash
ctest --test-dir build
```

---

## Choosing CUDA architectures

By default:

- CMake 3.24+ uses `native`
- older CMake falls back to the repository's explicit architecture list

If you want to override it, use CMake's native variable:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
```

Quick local Makefile flow:

```bash
make GPU_ARCH=sm_86
make benchmark
make test
```

---

## Validation boundary

| Environment | What to run |
|-------------|-------------|
| Local GPU machine | benchmark, runtime verification, `ctest` |
| Hosted CI | formatting, compile validation, OpenSpec/repository checks, Pages |

This split is intentional: GitHub-hosted runners validate repository health, while performance and CUDA runtime correctness still require a real GPU machine.

---

## Useful commands

```bash
# one explicit benchmark case
./build/bin/sgemm_benchmark --dims 256 384 640

# longer benchmark run
./build/bin/sgemm_benchmark -a --warmup 10 --benchmark 50

# OpenSpec validation
openspec validate --all
```

---

## Where to go next

- [Learning Path](learning-path/)
- [Architecture](architecture/)
- [Benchmark Results](benchmark-results/)
