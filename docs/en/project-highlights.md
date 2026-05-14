---
title: Project Highlights
---

# Project Highlights

This page is now a quick orientation surface. The full “why this design exists” narrative lives in the [Architecture section](/en/architecture/).

## What this repository is trying to prove

- SGEMM optimization can be explained as a reasoning chain, not a list of tricks.
- Performance claims are only trustworthy when tied to correctness policy and benchmark scope.
- Tensor Core acceleration is valuable only when its constraints and fallback behavior are made explicit.

## Best pages for each question

| If you want to know... | Start here |
|------------------------|------------|
| Why the whole system is structured this way | [Architecture Overview](/en/architecture/) |
| Why the kernels appear in this order | [Kernel Ladder](/en/architecture/kernel-ladder) |
| How data moves through memory and shared tiles | [Memory Flow](/en/architecture/memory-flow) |
| When WMMA is used and when it is rejected | [Tensor Core Path](/en/architecture/tensor-core-path) |
| How to present the project in interviews | [Interview Playbook](/en/interview-playbook) |

## Quick evaluators' path

1. [Architecture Overview](/en/architecture/)
2. [Kernel Ladder](/en/architecture/kernel-ladder)
3. [Benchmark Results](/en/benchmark-results)
4. [Interview Playbook](/en/interview-playbook)
5. [References](/en/references)
