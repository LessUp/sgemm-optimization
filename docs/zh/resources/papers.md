---
title: 相关论文与研究
---

# 相关论文与研究

本页将项目中的设计决策追溯到学术基础。每条引用将内核优化或架构选择与其理论或实证来源联系起来。

## 内存层次优化

这些论文解释了为什么分块和瓦片化是矩阵乘法性能的基础。

<Citation
  citeKey="Goto2008"
  title="Anatomy of High-Performance Matrix Multiplication"
  authors="Kazushige Goto, Robert A. van de Geijn"
  year="2008"
  venue="ACM TOMS"
  doi="10.1145/1391989.1391995"
/>

理解矩阵乘法性能为何由内存层次主导的基础论文。本项目的内核阶梯遵循相同的分块哲学，适配到 CUDA 的共享内存和寄存器堆层次结构。

<Citation
  citeKey="Hong2012"
  title="GPU Performance Optimization: A Case Study with Matrix Multiplication"
  authors="Taesoo Hong, Hyesoon Kim, Sang-Woo Park"
  year="2012"
  venue="IEEE TPDS"
  doi="10.1109/TPDS.2012.279"
/>

针对 GPU 的 GEMM 优化研究。有助于理解 CUDA 的执行模型如何改变分块策略，与 CPU BLAS 形成对比。

## Bank 冲突消除

这些来源解释了共享内存 bank 冲突问题以及本项目使用的填充解决方案。

<Citation
  citeKey="Ruetsch2009"
  title="Optimizing Matrix Multiply on GPUs"
  authors="Gregory Ruetsch, Massimiliano Fatica"
  year="2009"
  venue="GPU Computing Gems"
/>

共享内存 bank 冲突消除的实践方法。[消除 Bank Conflict 内核](/zh/kernel-bank-free)中的填充策略遵循此方法。

<Citation
  citeKey="Nvidia2007"
  title="CUDA Programming Guide: Shared Memory"
  authors="NVIDIA Corporation"
  year="2007"
  url="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory"
/>

共享内存 bank 冲突、访问模式和现代 GPU 上 32-bank 架构的官方文档。

## 双缓冲与延迟隐藏

这些来源解释了双缓冲内核中使用的重叠策略。

<Citation
  citeKey="Harris2007"
  title="Optimizing Parallel Reduction in CUDA"
  authors="Mark Harris"
  year="2007"
  venue="NVIDIA Developer Technology"
  url="https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf"
/>

虽然聚焦于归约操作，但此白皮书引入了双缓冲概念，用于重叠内存传输与计算。[双缓冲内核](/zh/kernel-double-buffer)将其应用于 GEMM 的瓦片加载-计算周期。

## Tensor Core 与混合精度

这些来源解释了 WMMA API 和混合精度性能特征。

<Citation
  citeKey="Jia2018"
  title="Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking"
  authors="Zhe Jia, Marco Maggioni, Jeffrey Smith, Daniele P. Scarpazza"
  year="2018"
  venue="arXiv"
  doi="10.48550/arXiv.1804.06826"
/>

Volta Tensor Core 的微基准测试研究。有助于理解 [Tensor Core WMMA 内核](/zh/kernel-tensor-core)背后的实际吞吐量和延迟特征。

<Citation
  citeKey="Nvidia2017"
  title="Programming Tensor Cores in CUDA 9"
  authors="NVIDIA Corporation"
  year="2017"
  url="https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/"
/>

WMMA API 的官方介绍。这是本项目中使用的片段类型、形状约束和混合精度语义的主要参考。

## 性能建模

这些来源提供了理解基准测试结果的理论框架。

<Citation
  citeKey="Williams2009"
  title="Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures"
  authors="Samuel Williams, Andrew Waterman, David Patterson"
  year="2009"
  venue="CACM"
  doi="10.1145/1498775.1498785"
/>

原始 Roofline 模型论文。提供了讨论算术强度、内存带宽限制和计算上限的词汇，这是 [Benchmark 纪律](/zh/methodology/benchmark-discipline)的基础。

## 如何使用本页

1. **阅读内核前**：打开对应的引用以理解优化原理。
2. **阅读内核后**：使用引用检查你的心智模型是否与已发表的解释匹配。
3. **面试准备**：这些引用为解释每个优化为何有效提供了学术支撑。

## BibTeX 导出

用于 LaTeX 文档或学术写作：

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

## 下一步

- [参考资料清单](/zh/references) — 文档、工具和代码库的完整目录
- [延伸阅读路线](/zh/resources/further-reading) — 有观点的学习路径
- [资源中心](/zh/resources/) — 基于场景的入口点
