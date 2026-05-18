---
title: Benchmark Discipline
---

# Benchmark Discipline

Benchmark discipline is about designing experiments that teach you something.

This repository already gives you a useful split: a single-shape run for isolation, a mixed-shape default set for regression hunting, and longer runs for publication-quality measurements. Use those modes deliberately instead of treating every benchmark as the same kind of evidence.

## Start with a hypothesis, not a command

Before you run anything, write down three things:

1. **The shape you are testing**
2. **The bottleneck you think you are testing**
3. **The one code change that should move that bottleneck**

If you cannot state those three items, the run is still exploration and should not be reported as a result.

## Canonical experiment templates

### Isolate one shape

```bash
./build/bin/sgemm_benchmark --dims 1024 1024 1024
```

Use this when you want to remove shape diversity and look at one bottleneck in a stable context.

### Sweep the repository default set

```bash
./build/bin/sgemm_benchmark -a
```

The default set intentionally mixes:

- aligned square cases: `512`, `1024`
- one aligned non-square case: `256 x 384 x 640`
- one unaligned edge case: `511 x 513 x 1025`

Use this when you want to learn whether a change is robust or only wins on friendly dimensions.

### Strengthen measurement confidence

```bash
./build/bin/sgemm_benchmark -a --warmup 10 --benchmark 50
```

Defaults are `--warmup 5 --benchmark 20`. Increase them before writing documentation, a PR summary, or any claim that compares kernels publicly.

## Shape selection rules

| Situation | Prefer | Why |
|----------|--------|-----|
| You are debugging a single regression | One explicit `--dims` case | Reduces noise and speeds iteration |
| You are checking whether a launch change generalizes | `-a` | Exposes aligned and unaligned behavior together |
| You are discussing WMMA | At least one 16-aligned case plus one unaligned case | Shows both fast-path potential and fallback consequences |
| You are documenting a final number | Canonical shape + irregular shape | Prevents a single friendly dimension from becoming a universal claim |

## Measurement rules that keep experiments honest

- **Keep one hypothesis per loop.** If you change tile shape, staging depth, and fallback policy together, the benchmark teaches nothing.
- **Do not skip correctness re-checks.** `ctest --test-dir build` is part of the loop, not post-processing.
- **Label the benchmark scope immediately.** Decide whether you are observing end-to-end behavior or a narrower compute-only path, then send readers to [Benchmark Scope](/en/validation/benchmark-scope) for interpretation.
- **Record the environment with the run.** GPU model, CUDA version, dimensions, warmup count, and benchmark count should travel with the result.

## Pre-publication checklist

- Run `ctest --test-dir build`
- Compare one canonical and one irregular shape
- Confirm whether the result is end-to-end or compute-only
- Capture the exact benchmark command
- Hand the result to [Validation](/en/validation/) before treating it as a repository claim
