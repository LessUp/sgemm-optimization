---
title: Curated References
---

# Curated References

This is the detailed catalog behind the [Resources Hub](/en/research/). The goal is not to dump links; it is to show which source helps with which kind of SGEMM question.

## How to use this page

- Start with the [Resources Hub](/en/research/) if you need help choosing a route.
- Use this page when you already know the category of source you need.
- Continue to [Further Reading Routes](/en/research/further-reading) when the right next topic matters more than the right citation.

## Official CUDA and NVIDIA docs

These are the sources to open when you need precise constraints, terminology, or API behavior.

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)  
  Best for execution model, synchronization, memory hierarchy, and launch semantics.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)  
  Best for optimization heuristics, memory access advice, and profiling-oriented sanity checks.
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)  
  Best for stream, event, launch, and runtime behavior details when the implementation questions get concrete.
- [CUDA Programming Guide: WMMA section](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)  
  Best for fragment types, shape constraints, and the mechanics behind Tensor Core usage.
- [NVIDIA Developer Blog: Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)  
  Best for a high-level explanation of why WMMA programming looks different from scalar CUDA code.
- [NVIDIA Mixed-Precision Training Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)  
  Best for understanding where mixed-precision wins come from and what extra constraints or conversion costs appear.

Why this shelf matters:

- It anchors whitepaper claims in vendor-defined behavior instead of community folklore.
- It helps you explain why unsupported shapes, alignment limits, and fallback rules are engineering constraints, not arbitrary policy.

## Papers and performance mental models

Open this shelf when you want the design logic behind SGEMM optimization rather than raw API detail.

- [Anatomy of High-Performance Matrix Multiplication (GotoBLAS paper)](https://www.cs.utexas.edu/users/flame/pubs/GotoTOMS_revision.pdf)  
  The best first paper here for understanding why blocking and hierarchy dominate matrix multiplication performance.
- [BLIS papers and project entry point](https://github.com/flame/blis)  
  Useful when you want to compare teaching-oriented kernels with a production CPU framework built around explicit packing and control trees.
- [Nsight Compute roofline charts guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts)  
  Useful when you need a stronger language for arithmetic intensity and the boundary between memory-bound and compute-bound behavior.

Why this shelf matters:

- It gives readers a mental model for why the kernel ladder is ordered the way it is.
- It helps you discuss performance ceilings without reducing everything to one benchmark number.

## Exemplary codebases and production-grade samples

Open this shelf when you want to compare this repository's explanatory style with industrial-strength implementations.

- [CUTLASS: Fast Linear Algebra in CUDA C++](https://github.com/NVIDIA/cutlass)  
  Best for seeing how a production CUDA GEMM library structures tiling, pipelines, and architecture-specific specialization.
- [BLIS Framework](https://github.com/flame/blis)  
  Best for understanding how GEMM decomposition and packing ideas generalize beyond CUDA.
- [CUDA Samples: matrixMul example](https://github.com/NVIDIA/cuda-samples/tree/master/cpp/0_Introduction/matrixMul)  
  Best for a smaller official example that is easier to compare with this project's early kernel stages.

Why this shelf matters:

- It shows where this repository is deliberately simplified for teaching.
- It gives interview follow-up material when someone asks what the "next production step" would look like.

## Profiler, tooling, and diagnosis references

Open this shelf when a benchmark number stops being self-explanatory and you need evidence.

- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)  
  Best for kernel-level counters, memory behavior, roofline views, and occupancy analysis.
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)  
  Best for timeline-level reasoning, launch gaps, overlap, and host/device interaction.
- [CUDA Occupancy Calculator (archived official workbook)](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-occupancy-calculator/index.html)  
  Best for turning block size, shared memory, and register usage into an explicit occupancy trade-off discussion.
- [Compute Sanitizer Documentation](https://docs.nvidia.com/compute-sanitizer/)  
  Best for catching correctness and memory issues before performance conclusions become trustworthy.

Why this shelf matters:

- It supports metric-driven diagnosis instead of guesswork.
- It connects directly to the repository's [Diagnosis Loop](/en/academy/diagnosis-loop), [Benchmark Scope](/en/validation/benchmark-scope), and [Reproducibility](/en/validation/reproducibility) pages.

## Engineering workflow and validation discipline

Open this shelf when the question is about proving correctness, structuring builds, or keeping claims reproducible.

- [GoogleTest Documentation](https://google.github.io/googletest/)  
  Best for understanding the local correctness harness and tolerance-oriented testing vocabulary.
- [CMake Documentation](https://cmake.org/documentation/)  
  Best for build-system expectations, generator behavior, and reproducible local setup.
- [OpenSpec documentation](https://openspec.dev/)  
  Best for understanding the spec-governed documentation and change workflow used in this repository.

Why this shelf matters:

- It reinforces the repository's local-GPU versus hosted-CI boundary model.
- It explains why "performance proof" and "repository integrity" live on different evidence surfaces.

## Next-step study routes

Use these when you know you want to keep learning, but not which topic should come first.

- [Further Reading Routes](/en/research/further-reading) for curated paths on tiling, occupancy, roofline thinking, Tensor Core constraints, and profiling.
- [CUDA Memory Cheat Sheet](/en/academy/cuda-memory-cheatsheet) for a fast memory refresher before reopening kernel code.
- [Resources Hub](/en/research/) for scenario-based entry points back into the rest of the site.
