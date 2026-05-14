---
title: 参考资料清单
---

# 参考资料清单

这是 [资源中心](/zh/resources/) 背后的详细目录。目标不是堆链接，而是说明：面对不同的 SGEMM 问题，应该先打开哪类资料。

## 这页怎么用

- 如果你还不知道该走哪条路线，先去 [资源中心](/zh/resources/)。
- 如果你已经知道自己要找“官方文档 / 论文 / profiler / 示例仓库”中的哪一类资料，就直接用本页。
- 如果你更关心“下一步先学什么”，而不是“该引用哪份文档”，就继续看 [延伸阅读路线](/zh/resources/further-reading)。

## CUDA 与 NVIDIA 官方文档

当你需要精确定义、约束边界或 API 行为时，先打开这一组。

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)  
  最适合核对执行模型、同步语义、内存层级与 launch 规则。
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)  
  最适合理解优化启发、内存访问建议与 profiler 前的常识性检查。
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)  
  当实现问题落到 stream、event、launch 或 runtime 细节时最有用。
- [CUDA Programming Guide: WMMA section](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)  
  最适合核对 fragment 类型、shape 约束与 Tensor Core 的调用机制。
- [NVIDIA Developer Blog: Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)  
  最适合快速理解 WMMA 编程为什么与普通 CUDA kernel 不同。
- [NVIDIA Mixed-Precision Training Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)  
  最适合理解混合精度收益从哪里来，以及它会带来哪些额外前提和转换成本。

为什么这组重要：

- 它让白皮书中的关键说法建立在厂商定义之上，而不是经验传说。
- 它帮助你解释：shape 限制、对齐要求、fallback 规则并不是“项目随手定的”，而是由底层约束推出来的。

## 论文与性能心智模型

当你想理解 SGEMM 优化背后的设计逻辑，而不是只查 API 时，打开这一组。

- [Anatomy of High-Performance Matrix Multiplication (GotoBLAS paper)](https://www.cs.utexas.edu/users/flame/pubs/GotoTOMS_revision.pdf)  
  最值得优先读，用来理解为什么 blocking 与层级化数据复用是矩阵乘法性能的核心。
- [BLIS papers and project entry point](https://github.com/flame/blis)  
  当你想比较教学型 kernel 与生产级 CPU GEMM 框架时非常有价值。
- [Nsight Compute roofline charts guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts)  
  当你需要更严谨地讨论算术强度，以及“内存受限 / 计算受限”的边界时很有帮助。

为什么这组重要：

- 它解释了为什么 kernel 阶梯是按现在这个顺序组织的。
- 它帮助你在讨论性能上限时，不会把所有东西都压缩成一个 benchmark 数字。

## 示范仓库与生产级样例

当你想对照成熟实现，看看本仓库在哪些地方是教学化简版时，打开这一组。

- [CUTLASS: Fast Linear Algebra in CUDA C++](https://github.com/NVIDIA/cutlass)  
  最适合观察生产级 CUDA GEMM 库如何组织 tiling、pipeline 和架构特化。
- [BLIS Framework](https://github.com/flame/blis)  
  最适合理解 GEMM 分解、packing 与控制树思想如何跨平台迁移。
- [CUDA Samples: matrixMul example](https://github.com/NVIDIA/cuda-samples/tree/master/cpp/0_Introduction/matrixMul)  
  官方、小而直观，适合与本项目早期 kernel 阶段对照阅读。

为什么这组重要：

- 它让你看清本仓库哪些地方是为了讲清楚概念而有意简化。
- 它能支撑“如果要继续工程化，下一步会长什么样”的回答。

## Profiler、工具与诊断资料

当 benchmark 数字本身已经不够说明问题，而你需要证据时，打开这一组。

- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)  
  最适合看 kernel 级计数器、内存行为、roofline 视图与 occupancy 分析。
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)  
  最适合看时间线、launch 间隙、重叠关系与 host/device 交互。
- [CUDA Occupancy Calculator (archived official workbook)](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-occupancy-calculator/index.html)  
  最适合把 block size、shared memory、寄存器使用量转换成显式的 occupancy 权衡讨论。
- [Compute Sanitizer Documentation](https://docs.nvidia.com/compute-sanitizer/)  
  最适合在讨论性能前，先把正确性与内存错误排干净。

为什么这组重要：

- 它支持基于指标的诊断，而不是拍脑袋解释。
- 它与仓库里的 [诊断闭环](/zh/methodology/diagnosis-loop)、[Benchmark 范围](/zh/validation/benchmark-scope)、[可复现性](/zh/validation/reproducibility) 直接相连。

## 工程流程与验证纪律

当问题转向“如何证明正确”“如何组织构建与复现”时，打开这一组。

- [GoogleTest Documentation](https://google.github.io/googletest/)  
  对理解本地正确性 harness 与容差测试话语体系最有帮助。
- [CMake Documentation](https://cmake.org/documentation/)  
  对构建系统预期、generator 行为和可复现本地环境最有帮助。
- [OpenSpec documentation](https://openspec.dev/)  
  对理解本仓库采用的 spec 驱动文档与变更流程最有帮助。

为什么这组重要：

- 它强化了“本地 GPU 验证”和“托管 CI 验证”是两种不同证据面的事实。
- 它解释了为什么“性能证明”和“仓库完整性”不会被混写在同一套结论里。

## 下一步学习路线

当你已经确定想继续学，但还不确定先学哪个主题时，从这里继续。

- [延伸阅读路线](/zh/resources/further-reading)：按主题整理好的学习路线，覆盖 tiling、occupancy、roofline、Tensor Core 约束与 profiling。
- [CUDA 内存速查表](/zh/cuda-memory-cheatsheet)：在重新打开 kernel 代码前，快速回忆关键内存问题。
- [资源中心](/zh/resources/)：按场景重新进入站内其他页面。
