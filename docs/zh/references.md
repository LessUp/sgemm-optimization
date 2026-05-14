---
title: 参考文献
---

# 参考文献

本页将项目中的关键设计选择映射到权威技术资料，便于追溯和延展学习。

## CUDA 与 GPU 基础

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)

为什么重要：

- 为所有 kernel 阶段的执行模型假设提供官方定义。
- 让内存访问与同步讨论使用统一术语和边界。

## Tensor Core 与 WMMA

- [NVIDIA WMMA API Reference](https://docs.nvidia.com/cuda/cuda-c-api/group__CUDA__WMMA.html)
- [NVIDIA Developer Blog: Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- [NVIDIA Mixed-Precision Training Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)

为什么重要：

- 对应 WMMA 片段、对齐约束、混合精度行为等核心问题。
- 解释为什么非友好 shape 需要显式 fallback 策略。

## GEMM 优化研究与方法论

- [Anatomy of High-Performance Matrix Multiplication (GotoBLAS paper)](https://www.cs.utexas.edu/users/flame/pubs/GotoTOMS_revision.pdf)
- [CUTLASS: Fast Linear Algebra in CUDA C++](https://github.com/NVIDIA/cutlass)
- [BLIS Framework](https://github.com/flame/blis)

为什么重要：

- 将本项目的分阶段优化思路放到更广义 GEMM 方法论中理解。
- 为“如果做生产化下一步怎么做”提供高质量延展依据。

## Profiling 与性能分析

- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/)

为什么重要：

- 支持从单点 GFLOPS 走向指标驱动诊断。
- 帮助解释瓶颈归因、调优权衡与收益来源。

## 工程流程与验证纪律

- [GoogleTest Documentation](https://google.github.io/googletest/)
- [CMake Documentation](https://cmake.org/documentation/)
- [OpenSpec Documentation](https://github.com/openspec-ai/openspec)

为什么重要：

- 让仓库中的“正确性验证”和“流程治理”有权威工具链支撑。
- 强化本地 GPU 与托管 CI 验证边界的合理性。

## 相关页面

- [架构概述](/zh/architecture)
- [Benchmark 结果](/zh/benchmark-results)
- [优化手册](/zh/optimization-playbook)
