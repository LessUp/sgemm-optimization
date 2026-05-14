---
title: Correctness Policy
---

# Correctness Policy

Performance work is only accepted when correctness still holds against the repository's oracle.

## Oracle and baseline

The project uses **cuBLAS SGEMM** as the reference implementation for runtime verification. Kernel outputs are compared against that reference rather than against another project-local kernel.

This matters because optimization work frequently changes launch geometry, staging, and mixed-precision behavior. A local baseline is not strong enough for those checks.

## Tolerance policy

The verifier uses a NumPy-style `allclose` rule:

```text
|test - ref| <= atol + rtol * |ref|
```

### Standard FP32 kernels

- `rtol = 1e-3`
- `atol = 1e-4`

These thresholds apply to the Naive, Tiled, Bank Conflict Free, and Double Buffer paths.

### Tensor Core / mixed precision path

- `rtol = 5e-2`
- `atol = 1e-2`

The Tensor Core path uses a relaxed tolerance because it includes mixed-precision behavior.

## Shape coverage expectations

The test suite covers more than one friendly square case.

- standard dimensions include tiny, square, rectangular, and irregular shapes
- Tensor Core fast-path dimensions include 16-aligned cases
- Tensor Core fallback dimensions include deliberately unaligned cases such as `15x15x15`, `17x19x23`, and `511x513x1025`

A performance change is not considered trustworthy if it only works on the friendly subset while silently weakening the fallback path.

## CI versus local responsibility

Hosted CI can prove structure, documentation workflows, and governance alignment. It does **not** prove runtime correctness for CUDA kernels.

Local GPU execution must run:

```bash
ctest --test-dir build
```

Only after that passes should a benchmark-based performance statement be considered reviewable.
