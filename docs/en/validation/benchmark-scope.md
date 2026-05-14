---
title: Benchmark Scope
---

# Benchmark Scope

This page defines what the benchmark labels mean and how readers should interpret the reported numbers.

## Canonical label split

The benchmark suite distinguishes two Tensor Core views:

- **WMMA end-to-end**: the safe FP32-facing path, including FP32→FP16 conversion and fallback behavior
- **WMMA compute-only**: the pure pre-converted WMMA path, shown only when `M`, `K`, and `N` are positive multiples of 16

Those labels are intentionally different because they answer different questions.

## What each label proves

| Label | What it proves | What it cannot justify on its own |
|------|-----------------|-----------------------------------|
| cuBLAS | Reference throughput on the current GPU and toolchain | A direct statement about project kernel design quality across all environments |
| Standard FP32 kernels | End-to-end behavior of the repository's FP32 path on the chosen shapes | Any claim about Tensor Core potential |
| WMMA end-to-end | What a real caller experiences through the repository's safe Tensor Core wrapper | Peak Tensor Core compute throughput |
| WMMA compute-only | The upper bound of the pure WMMA compute path on compatible dimensions | The cost of conversion, fallback, or irregular shapes |

## How readers should interpret reported numbers

1. **Treat them as representative, not universal.** A published snapshot documents one GPU, one CUDA stack, and one benchmark configuration.
2. **Compare like with like.** Never compare aligned-only compute-only numbers to mixed-shape end-to-end numbers without saying so.
3. **Expect hardware sensitivity.** Volta, Turing, Ampere, Ada, and Hopper will emphasize different bottlenecks.
4. **Assume CI did not produce the number.** Hosted CI proves repository health, not benchmark truth.

## Canonical benchmark sets

The CLI defaults are part of the trust story:

- `1024 x 1024 x 1024` is the default single-case fallback when no dimensions are given.
- `-a` expands to `512x512x512`, `1024x1024x1024`, `256x384x640`, and `511x513x1025`.

That mix exists so the repository can report both friendly and awkward shapes without pretending they are the same workload.

## Before you cite a number

- name the GPU and CUDA environment
- state the exact command
- state whether the number is end-to-end or compute-only
- state whether the shape set is one case or the mixed default set
- point readers to [Reproducibility](/en/validation/reproducibility) if they need to re-run it
