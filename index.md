---
layout: default
title: Home
nav_order: 1
has_children: true
permalink: /
description: A CUDA SGEMM optimization portal that turns matrix multiplication kernels into a readable path from baseline code to Tensor Core WMMA.
lang: en
page_key: home
lang_ref: zh-home
---

{: .hero-section }
# SGEMM Optimization
{: .hero-title }

Learn CUDA matrix multiplication by walking the exact path from one-thread-per-output SGEMM to guarded Tensor Core WMMA.
{: .hero-subtitle }

[Start the walkthrough](docs/getting-started/){: .btn .fs-5 .mb-4 .mb-md-0 }
[Follow the kernel ladder](docs/learning-path/){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }
[View on GitHub](https://github.com/LessUp/sgemm-optimization){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }

---

## Why this project exists

Most GEMM examples either hide the details inside a production library or stop at a toy kernel. This project keeps the middle ground: each optimization step is isolated, readable, benchmarkable, and verified against cuBLAS.

<div class="perf-grid">
  <div class="perf-card">
    <div class="perf-label">Path</div>
    <div class="perf-value">5 stages</div>
    <div class="perf-vs">baseline to WMMA</div>
  </div>
  <div class="perf-card">
    <div class="perf-label">Reference</div>
    <div class="perf-value">cuBLAS</div>
    <div class="perf-vs">correctness oracle</div>
  </div>
  <div class="perf-card">
    <div class="perf-label">Interface</div>
    <div class="perf-value">1 shape</div>
    <div class="perf-vs">swap kernels cleanly</div>
  </div>
</div>

---

## Optimization ladder

| Stage | Kernel | The question it answers |
|------:|--------|-------------------------|
| 1 | [Naive](docs/kernel-naive/) | What is the simplest correct GPU mapping? |
| 2 | [Tiled](docs/kernel-tiled/) | How much does shared-memory reuse change the cost model? |
| 3 | [Bank-Free](docs/kernel-bank-free/) | Why does shared-memory layout matter after tiling? |
| 4 | [Double Buffer](docs/kernel-double-buffer/) | How do staged tiles hide global-memory latency? |
| 5 | [Tensor Core](docs/kernel-tensor-core/) | Where does WMMA help, and when should the wrapper fall back? |

---

## How the repository earns trust

| Concern | Project answer |
|---------|----------------|
| Numerical correctness | Google Test cases compare kernels with cuBLAS using FP32 and mixed-precision tolerances. |
| Benchmark honesty | The benchmark prints cuBLAS, FP32 kernels, Tensor Core end-to-end, and compute-only WMMA separately. |
| Unsupported shapes | The public Tensor Core wrapper uses a guarded fallback for non-16-aligned dimensions. |
| Hosted CI limits | CI validates formatting, CUDA compilation, OpenSpec structure, and Pages; runtime tests remain local GPU work. |

---

## Choose your route

| If you want to... | Start here |
|-------------------|------------|
| Build and run once | [Getting Started](docs/getting-started/) |
| Learn in the intended order | [Learning Path](docs/learning-path/) |
| Understand file boundaries | [Architecture](docs/architecture/) |
| Interpret benchmark output | [Benchmark Results](docs/benchmark-results/) |
| Inspect stable requirements | [Specifications Index](specs/) |

---

## Repository map

```text
src/kernels/   five SGEMM implementations
src/utils/     CUDA RAII, verification, benchmark helpers
tests/         Google Test coverage against cuBLAS
docs/          English learning path
zh/docs/       Chinese learning path
openspec/      stable specs and change workflow
```

---

## Explore next

[Getting Started](docs/getting-started/){: .btn .mr-2 }
[Learning Path](docs/learning-path/){: .btn .btn-outline .mr-2 }
[中文首页](zh/){: .btn .btn-outline }
