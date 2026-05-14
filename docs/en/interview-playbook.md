---
title: Interview Playbook
---

# Interview Playbook

Use this page to present the project with clarity in technical interviews.

## 90-second story

> I built this SGEMM project as an engineering notebook rather than a one-off benchmark script.  
> It starts from a readable FP32 baseline and moves through tiled, bank-conflict-free, double-buffered, and Tensor Core WMMA paths.  
> Each step addresses one bottleneck class and keeps the same launcher contract for comparability.  
> Correctness is always checked against cuBLAS, with separate tolerances for standard FP32 and mixed-precision WMMA.  
> I also separate end-to-end and compute-only measurements to avoid misleading Tensor Core claims.  
> The result is not “beating cuBLAS,” but demonstrating trustworthy optimization reasoning and engineering discipline.

## Recommended narrative structure

1. **Problem framing**  
   Explain why SGEMM is a useful proxy for memory hierarchy, parallel mapping, and mixed-precision trade-offs.

2. **Optimization ladder**  
   Walk through what each kernel changes and why that change should improve performance.

3. **Correctness and trust model**  
   Explain cuBLAS oracle checks, tolerance differences, and fallback behavior for non-16-aligned WMMA inputs.

4. **Measurement discipline**  
   Clarify end-to-end vs compute-only numbers and local GPU vs hosted CI boundaries.

5. **Engineering maturity**  
   Mention unified interfaces, RAII/error handling, bilingual docs, and OpenSpec governance.

## Deep-dive questions and high-quality answer patterns

### Q1: Why not just use cuBLAS and stop there?

Strong answer pattern:

- For production, yes, use cuBLAS/CUTLASS.
- This project is for learning and demonstration of performance reasoning.
- The value is showing I can diagnose bottlenecks and validate claims, not replacing vendor libraries.

### Q2: Why is Tensor Core only around a fraction of cuBLAS in your numbers?

Strong answer pattern:

- My WMMA path is intentionally educational, not fully production-optimized.
- cuBLAS uses deeper tiling, scheduling, epilog fusion, and architecture-specific kernels.
- I explicitly report compute-only and end-to-end to show where overhead appears.

### Q3: How do you ensure performance claims are trustworthy?

Strong answer pattern:

- Numerical correctness is checked before discussing speedups.
- I report benchmark scope and shape conditions.
- I separate CI-safe checks from GPU-required runtime checks.
- I keep fallback behavior explicit for unsupported WMMA dimensions.

### Q4: What would you improve next if this became production work?

Strong answer pattern:

- Add architecture-specific launch tuning.
- Improve WMMA staging and overlap depth.
- Expand profiler-driven evidence (Nsight Compute metrics).
- Evaluate CUTLASS-based custom kernels for maintainable high performance.

## Common interview mistakes to avoid

- Claiming “Tensor Core is always faster” without shape and conversion caveats.
- Presenting one GFLOPS number without measurement scope.
- Ignoring numerical tolerance implications in mixed precision.
- Mixing CI build success with runtime correctness claims.

## Fast prep checklist

- [ ] Can explain each optimization stage in one sentence.
- [ ] Can explain why FP32 and WMMA tolerances differ.
- [ ] Can explain why benchmark numbers have multiple labels.
- [ ] Can explain what CI validates vs what local GPU validates.
- [ ] Can explain why this project is useful even if it does not beat cuBLAS.

## Related pages

- [Architecture Overview](/en/architecture/)
- [Kernel Ladder](/en/architecture/kernel-ladder)
- [Benchmark Results](/en/benchmark-results)
- [Learning Path](/en/learning-path)
- [References](/en/references)
