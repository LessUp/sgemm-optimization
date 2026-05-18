---
title: Project Guide
---

# Project Guide

This is the executive guide for the whitepaper site. Use it when you need the repository’s positioning, reader map, and recommended reading order before opening the deeper sections.

## What this project is

This repository is a CUDA SGEMM study designed around a staged kernel ladder:

1. readable FP32 baseline
2. shared-memory tiling
3. bank-conflict mitigation
4. double-buffer overlap
5. guarded Tensor Core WMMA

The goal is not to beat production GEMM libraries. The goal is to show how an optimization argument is built, defended, and limited.

## Who this site is for

| Reader | What to open first | Why |
|---|---|---|
| Interviewer | [Architecture](../architecture/) | Fastest way to audit system clarity and differentiation |
| Candidate preparing a walkthrough | [Academy](../academy/) | Gives a teaching order and interview-safe explanations |
| CUDA learner | [Architecture](../architecture/) then [Academy](../academy/) | Preserves the conceptual ladder before code detail |
| Performance skeptic | [Validation](../validation/) | Shows where proof begins and where it stops |
| Research-minded reader | [Research](../research/) | Adds papers, related repos, and lineage |

## How the site is structured

| Section | Primary job |
|---|---|
| [Overview](./) | Orientation and reading strategy |
| [Architecture](../architecture/) | System map, bottlenecks, invariants |
| [Academy](../academy/) | Ordered study of the optimization ladder |
| [Validation](../validation/) | Correctness and benchmark trust boundary |
| [Research](../research/) | References, related work, and evolution notes |

## Fast reading plans

### Reviewer path

1. [Architecture Overview](../architecture/)
2. [Kernel Ladder](../architecture/kernel-ladder)
3. [Validation Overview](../validation/)
4. [Related Projects](../research/related-projects)

### Candidate path

1. [Academy Overview](../academy/)
2. [Learning Path](../academy/learning-path)
3. [Diagnosis Loop](../academy/diagnosis-loop)
4. [Evolution Notes](../research/evolution)

### Builder path

1. [Getting Started](./getting-started)
2. [Correctness Policy](../validation/correctness-policy)
3. [Benchmark Scope](../validation/benchmark-scope)
4. [Curated References](../research/references)
