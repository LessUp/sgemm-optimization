---
title: Project Highlights
---

# Project Highlights

This page explains why this repository stands out when compared with many SGEMM demos.

## What makes it differentiated

| Dimension | Typical SGEMM demo | This repository |
|-----------|---------------------|-----------------|
| Learning structure | One or two kernels without clear progression | Five-stage kernel ladder with explicit learning intent |
| Correctness discipline | Spot-check outputs or no clear oracle | cuBLAS-backed verification with separate FP32/WMMA tolerance policy |
| Performance claims | Single number without context | End-to-end and compute-only labels, plus scope boundaries |
| Engineering governance | Docs and code can drift | OpenSpec-driven alignment for docs, workflow, and requirements |
| Interview readiness | Hard to narrate as engineering story | Dedicated interview playbook and proof-first homepage |

## Strengths that interviewers usually value

### 1) Clear decision chain

The optimization path is not random tuning. It is a sequence of explicit bottlenecks:

- Naive: establish baseline and expose memory bottlenecks
- Tiled: introduce shared-memory reuse
- Bank-Free: reduce bank-conflict penalties with padded layouts
- Double Buffer: overlap memory and compute
- Tensor Core: raise throughput ceiling with mixed precision and guarded fallback

### 2) Evidence over slogans

Performance and correctness are coupled in public storytelling:

- Benchmark scope is labeled (`end-to-end` vs `compute-only`)
- Correctness policy is explicit (different tolerances for FP32 and WMMA)
- Validation boundaries are explicit (CI-safe checks vs local GPU runtime checks)

### 3) Practical engineering boundaries

The project documents what CI can and cannot prove:

- Hosted CI: formatting, compile validity, repository/spec integrity, Pages buildability
- Local GPU machine: runtime correctness verification and performance benchmarking

This boundary is useful in interviews because it demonstrates realistic engineering judgment.

## Repository-level quality signals

- Consistent kernel launcher contract for swappability
- RAII-based CUDA resource handling
- Exception-based error reporting
- Bilingual mirrored docs for public accessibility
- OpenSpec-based requirements and change workflow

## Recommended walkthrough order (for evaluators)

1. [Getting Started](/en/getting-started)
2. [Learning Path](/en/learning-path)
3. [Benchmark Results](/en/benchmark-results)
4. [Interview Playbook](/en/interview-playbook)
5. [References](/en/references)
