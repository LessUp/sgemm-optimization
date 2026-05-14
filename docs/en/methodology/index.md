---
title: Methodology
---

# Methodology

This section explains **how the repository is optimized**.

Architecture tells you why the kernels are arranged the way they are. Methodology tells you how to study that ladder, isolate bottlenecks, and turn one observation into one defensible change. Validation then tells you which claims survive correctness checks, scope labeling, and reproducibility review.

## What belongs here

| Topic | Why it lives in Methodology | Canonical page |
|------|------|------|
| Learning the kernel ladder in order | The reader needs a staged path before tuning details make sense | [Learning Path](/en/learning-path) |
| Running disciplined experiments | Experiment design is part of the optimization workflow | [Benchmark Discipline](/en/methodology/benchmark-discipline) |
| Diagnosing bottlenecks and choosing the next hypothesis | This is the core optimization loop | [Diagnosis Loop](/en/methodology/diagnosis-loop) |
| Per-kernel implementation detail | These pages explain what each optimization stage changes | [Kernel pages](/en/kernel-naive) |

## Methodology map

```mermaid
flowchart TD
    A[Architecture map] --> B[Learning path]
    B --> C[Benchmark discipline]
    C --> D[Diagnosis loop]
    D --> E[Minimal code change]
    E --> F[Validation review]
    F -->|still correct, properly labeled, reproducible| G[Keep and document]
    F -->|claim is weak or incorrect| H[Rollback or narrow the claim]
```

## Working rules

1. **Learn the ladder in order.** Start from the architectural map and the kernel progression before touching WMMA-specific conclusions.
2. **Change one thing at a time.** A single benchmark cycle should validate one hypothesis, not a bundle of unrelated edits.
3. **Use local GPU evidence early.** Diagnosis without runtime evidence usually misclassifies the bottleneck.
4. **Hand every speedup to Validation.** A gain only counts after correctness, scope, and reproducibility are re-checked.

## Read this section in order

1. [Learning Path](/en/learning-path)
2. [Benchmark Discipline](/en/methodology/benchmark-discipline)
3. [Diagnosis Loop](/en/methodology/diagnosis-loop)
4. [Validation Overview](/en/validation/)
