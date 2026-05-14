---
title: References
---

# References

This list maps project decisions to authoritative technical sources.

## CUDA and GPU fundamentals

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)

Why this group matters:

- Defines execution model assumptions used by all kernel stages.
- Anchors memory and synchronization discussions in official terminology.

## Tensor Core and WMMA

- [NVIDIA WMMA API Reference](https://docs.nvidia.com/cuda/cuda-c-api/group__CUDA__WMMA.html)
- [NVIDIA Developer Blog: Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- [NVIDIA Mixed-Precision Training Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)

Why this group matters:

- Supports WMMA fragment, alignment, and mixed-precision behavior discussion.
- Explains why fallback policies are necessary for non-friendly shapes.

## GEMM optimization research and methodology

- [Anatomy of High-Performance Matrix Multiplication (GotoBLAS paper)](https://www.cs.utexas.edu/users/flame/pubs/GotoTOMS_revision.pdf)
- [CUTLASS: Fast Linear Algebra in CUDA C++](https://github.com/NVIDIA/cutlass)
- [BLIS Framework](https://github.com/flame/blis)

Why this group matters:

- Connects this project's staged optimization mindset to broader GEMM methodology.
- Provides production-grade references for interview follow-up discussions.

## Profiling and performance analysis

- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/)

Why this group matters:

- Supports diagnosis beyond single GFLOPS outputs.
- Enables metric-driven explanations of bottlenecks and trade-offs.

## Engineering process and validation discipline

- [GoogleTest Documentation](https://google.github.io/googletest/)
- [CMake Documentation](https://cmake.org/documentation/)
- [OpenSpec Documentation](https://github.com/openspec-ai/openspec)

Why this group matters:

- Grounds the repository's correctness and workflow claims in established tooling.
- Reinforces the local-GPU vs hosted-CI validation boundary model.

## Related project pages

- [Architecture Overview](/en/architecture)
- [Benchmark Results](/en/benchmark-results)
- [Optimization Playbook](/en/optimization-playbook)
