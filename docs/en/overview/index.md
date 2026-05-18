---
title: Project Guide
---

# Project Guide

This is the orientation surface for the SGEMM whitepaper site. Use it to understand the project's positioning, intended audience, and recommended reading order before entering the deeper sections.

## What this project is

This repository is a CUDA SGEMM study organized around a five-stage kernel optimization ladder:

1. **Naïve FP32** — baseline cost model, no shared-memory reuse
2. **Tiled FP32** — shared-memory staging, arithmetic intensity rises with tile size
3. **Bank-Free FP32** — padding eliminates avoidable bank conflicts
4. **Double Buffer** — overlapped staging and compute hides memory latency
5. **Tensor Core WMMA** — hardware fragment accumulation, guarded by device capability and shape constraints

The goal is not to produce the fastest possible SGEMM implementation. The goal is to show how an optimization argument is built, bounded, and defended — in a form that is readable under interview pressure and auditable by an experienced CUDA engineer.

## Who this site is for

| Reader | Best first page | Time |
|---|---|---|
| Interviewer auditing system clarity | [Architecture Overview](../architecture/) | 8 min |
| Candidate preparing a walkthrough | [Academy Overview](../academy/) | 5 min then follow [Learning Path](../academy/learning-path) |
| CUDA learner starting fresh | Here, then [Architecture](../architecture/), then [Academy](../academy/) | Self-paced |
| Performance skeptic | [Validation Overview](../validation/) | 12 min |
| Research-minded reader | [Research Desk](../research/) | Self-paced |

See [Reader Map](./reader-map) for a full depth-tiered navigation index.

## How the site is structured

Each section has one primary job. This is deliberate: a page with two jobs is a page that does neither well.

| Section | Primary job | What it is not |
|---|---|---|
| [Overview](./) | Orientation and reading strategy | Not a replacement for the architecture section |
| [Architecture](../architecture/) | System map, bottlenecks, invariants | Not a code walkthrough |
| [Academy](../academy/) | Ordered study of the optimization ladder | Not a reference manual |
| [Validation](../validation/) | Correctness and benchmark trust boundary | Not a performance claim |
| [Research](../research/) | References, related work, evolution notes | Not an extended bibliography |

## Fast reading plans

### Reviewer path (20 min)

1. [Architecture Overview](../architecture/)
2. [Kernel Ladder](../architecture/kernel-ladder)
3. [Validation Overview](../validation/)
4. [Related Projects](../research/related-projects)

### Candidate path (30 min)

1. [Academy Overview](../academy/)
2. [Learning Path](../academy/learning-path)
3. [Diagnosis Loop](../academy/diagnosis-loop)
4. [Evolution Notes](../research/evolution)

### Builder path (self-paced)

1. [Getting Started](./getting-started)
2. [System Blueprint](../architecture/system-blueprint)
3. [Correctness Policy](../validation/correctness-policy)
4. [Benchmark Scope](../validation/benchmark-scope)
5. [Curated References](../research/references)

## What makes this site a whitepaper, not a portfolio

Most project documentation describes what was built. This site argues *why* each architectural decision was made, *what evidence* constrains the claims, and *where the reasoning stops*.

That distinction matters when an interviewer asks: "Why does the bank-free kernel exist? What does it actually improve?" A portfolio answer is: "It's faster." A whitepaper answer is: "Shared-memory bank conflicts serialize access when multiple threads map to the same bank. Padding the tile layout by one element shifts each column to a different bank, eliminating the multi-way conflict. The improvement is real on conflict-prone shapes and measurable on the test hardware; it is not universal."

That is the register this site aims for across all five kernel stages, across both architecture and validation surfaces.
