---
title: Related Papers & Research
---

# Related Papers & Research

This page traces the design decisions in this project back to academic foundations. Each citation connects a kernel optimization or architectural choice to its theoretical or empirical source.

## Memory Hierarchy Optimization

These papers explain why blocking and tiling are fundamental to matrix multiplication performance.

<Citation
  citeKey="Goto2008"
  title="Anatomy of High-Performance Matrix Multiplication"
  authors="Kazushige Goto, Robert A. van de Geijn"
  year="2008"
  venue="ACM TOMS"
  doi="10.1145/1391989.1391995"
/>

The foundational paper for understanding why matrix multiplication performance is dominated by memory hierarchy. This project's kernel ladder follows the same blocking philosophy, adapted to CUDA's shared memory and register file hierarchy.

<Citation
  citeKey="Hong2012"
  title="GPU Performance Optimization: A Case Study with Matrix Multiplication"
  authors="Taesoo Hong, Hyesoon Kim, Sang-Woo Park"
  year="2012"
  venue="IEEE TPDS"
  doi="10.1109/TPDS.2012.279"
/>

A GPU-specific treatment of GEMM optimization. Useful for understanding how CUDA's execution model changes the blocking strategy compared to CPU BLAS.

## Bank Conflict Avoidance

These sources explain the shared memory bank conflict problem and the padding solution used in this project.

<Citation
  citeKey="Ruetsch2009"
  title="Optimizing Matrix Multiply on GPUs"
  authors="Gregory Ruetsch, Massimiliano Fatica"
  year="2009"
  venue="GPU Computing Gems"
/>

Practical treatment of bank conflict avoidance in shared memory. The padding strategy in [Bank-Free Kernel](/en/academy/kernel-bank-free) follows this approach.

<Citation
  citeKey="Nvidia2007"
  title="CUDA Programming Guide: Shared Memory"
  authors="NVIDIA Corporation"
  year="2007"
  url="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory"
/>

Official documentation for shared memory bank conflicts, access patterns, and the 32-bank architecture on modern GPUs.

## Double Buffering and Latency Hiding

These sources explain the overlap strategy used in the double-buffer kernel.

<Citation
  citeKey="Harris2007"
  title="Optimizing Parallel Reduction in CUDA"
  authors="Mark Harris"
  year="2007"
  venue="NVIDIA Developer Technology"
  url="https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf"
/>

While focused on reduction, this whitepaper introduces the double-buffer concept for overlapping memory transfers with computation. The [Double Buffer Kernel](/en/academy/kernel-double-buffer) applies this to GEMM's tile load-compute cycle.

## Tensor Core and Mixed Precision

These sources explain the WMMA API and mixed-precision performance characteristics.

<Citation
  citeKey="Jia2018"
  title="Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking"
  authors="Zhe Jia, Marco Maggioni, Jeffrey Smith, Daniele P. Scarpazza"
  year="2018"
  venue="arXiv"
  doi="10.48550/arXiv.1804.06826"
/>

Microbenchmarking study of Volta Tensor Cores. Useful for understanding the actual throughput and latency characteristics behind the [Tensor Core WMMA Kernel](/en/academy/kernel-tensor-core).

<Citation
  citeKey="Nvidia2017"
  title="Programming Tensor Cores in CUDA 9"
  authors="NVIDIA Corporation"
  year="2017"
  url="https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/"
/>

Official introduction to the WMMA API. This is the primary reference for fragment types, shape constraints, and the mixed-precision semantics used in this project.

## Performance Modeling

These sources provide the theoretical framework for interpreting benchmark results.

<Citation
  citeKey="Williams2009"
  title="Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures"
  authors="Samuel Williams, Andrew Waterman, David Patterson"
  year="2009"
  venue="CACM"
  doi="10.1145/1498775.1498785"
/>

The original roofline model paper. Provides the vocabulary for discussing arithmetic intensity, memory bandwidth limits, and compute ceilings that underpins [Benchmark Discipline](/en/academy/benchmark-discipline).

## How to Use This Page

1. **Before reading a kernel**: Open the corresponding citation to understand the optimization principle.
2. **After reading a kernel**: Use the citation to check whether your mental model matches the published explanation.
3. **For interviews**: These citations provide the academic grounding for explaining why each optimization works.

## BibTeX Export

For LaTeX documents or academic writing, you can copy the following BibTeX entries:

```bibtex
@article{Goto2008,
  author    = {Kazushige Goto and Robert A. van de Geijn},
  title     = {Anatomy of High-Performance Matrix Multiplication},
  journal   = {ACM Transactions on Mathematical Software},
  year      = {2008},
  volume    = {34},
  number    = {3},
  doi       = {10.1145/1391989.1391995}
}

@article{Hong2012,
  author    = {Taesoo Hong and Hyesoon Kim and Sang-Woo Park},
  title     = {GPU Performance Optimization: A Case Study with Matrix Multiplication},
  journal   = {IEEE Transactions on Parallel and Distributed Systems},
  year      = {2012},
  volume    = {23},
  number    = {6},
  doi       = {10.1109/TPDS.2012.279}
}

@inbook{Ruetsch2009,
  author    = {Gregory Ruetsch and Massimiliano Fatica},
  title     = {Optimizing Matrix Multiply on GPUs},
  booktitle = {GPU Computing Gems},
  year      = {2009},
  publisher = {Morgan Kaufmann}
}

@article{Williams2009,
  author    = {Samuel Williams and Andrew Waterman and David Patterson},
  title     = {Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures},
  journal   = {Communications of the ACM},
  year      = {2009},
  volume    = {52},
  number    = {4},
  doi       = {10.1145/1498775.1498785}
}
```

## Next Steps

- [Curated References](/en/research/references) — Full catalog of documentation, tools, and codebases
- [Further Reading Routes](/en/research/further-reading) — Opinionated study paths
- [Resources Hub](/en/research/) — Scenario-based entry points
