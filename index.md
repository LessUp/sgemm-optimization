---
layout: default
title: Home
nav_order: 1
has_children: true
permalink: /
description: Bilingual CUDA SGEMM optimization tutorial and reference implementation, from naive kernels to Tensor Core WMMA.
lang: en
page_key: home
lang_ref: zh-home
---

{: .hero-section }
# SGEMM Optimization
{: .hero-title }

From readable baseline code to Tensor Core WMMA
{: .hero-subtitle }

[🚀 Start Here](docs/getting-started){: .btn .fs-5 .mb-4 .mb-md-0 }
[📚 Learning Path](docs/learning-path){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }

---

## Why this project is useful

This repository is designed to be a compact CUDA GEMM learning and reference project:

- **Progressive**: five kernel variants show what each optimization step changes
- **Verifiable**: every kernel is checked against cuBLAS
- **Practical**: benchmark and test entry points are already wired up
- **Maintainable**: repository rules, workflow, and validation are documented through OpenSpec

---

## Optimization ladder

| Stage | Kernel | What you learn |
|------:|--------|----------------|
| 1 | [Naive](docs/kernel-naive) | Thread-to-output mapping and baseline cost |
| 2 | [Tiled](docs/kernel-tiled) | Shared-memory blocking and data reuse |
| 3 | [Bank-Free](docs/kernel-bank-free) | Padding away 32-way bank conflicts |
| 4 | [Double Buffer](docs/kernel-double-buffer) | Latency hiding through staged tiles |
| 5 | [Tensor Core](docs/kernel-tensor-core) | WMMA usage with a guarded fallback path |

---

## What is inside the repository

| Surface | Purpose |
|---------|---------|
| `src/` | CUDA kernels, benchmark entry point, and utilities |
| `tests/` | Google Test verification against cuBLAS |
| `docs/` | Learning-oriented technical documentation |
| `openspec/` | Stable requirements, workflow, and change history |

---

## How to use it

### 1. Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/bin/sgemm_benchmark -a
ctest --test-dir build
```

### 2. Follow the learning route

- [Getting Started](docs/getting-started)
- [Learning Path](docs/learning-path)
- [Architecture](docs/architecture)
- [Benchmark Notes](docs/benchmark-results)

### 3. Inspect the project rules

- [Specifications Index](specs)
- [OpenSpec Workflow Notes](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/README.md)
- [Repository Guide](https://github.com/LessUp/sgemm-optimization/blob/master/AGENTS.md)

---

## Validation boundary

- **Local GPU machine**: runtime verification and benchmarking
- **Hosted CI**: formatting, compilation, OpenSpec/repository checks, and Pages

That split keeps the repository honest without pretending GitHub-hosted runners can replace a real CUDA runtime environment.

---

## Explore next

[📘 Benchmark results](docs/benchmark-results){: .btn .btn-outline .mr-2 }
[🏗️ Architecture overview](docs/architecture){: .btn .btn-outline .mr-2 }
[⭐ View on GitHub](https://github.com/LessUp/sgemm-optimization){: .btn .btn-outline }
