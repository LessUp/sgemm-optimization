---
title: 资源中心
---

# 资源中心

当读者看完架构与方法论后，真正的下一步不该只是碰运气地点开一串链接；这页就是为那个交接点准备的。

这里不是平铺式书单，而是带有编辑判断的资源入口：每条路线都会说明应该先看什么、为什么重要、以及它能回答项目中的哪类问题。

## 先按你手上的问题选入口

| 如果你的问题更像这样 | 先看这里 | 为什么这条路线有用 |
|---|---|---|
| “这个 kernel 为什么这么在意内存布局？” | [CUDA 内存速查表](/zh/cuda-memory-cheatsheet) | 先把合并访问、共享内存布局、Tensor Core 对齐和 benchmark 前该检查的点重新捡起来。 |
| “白皮书里的这些约束到底对应哪份官方文档？” | [参考资料清单](/zh/references#cuda-与-nvidia-官方文档) | 直接跳到 CUDA、WMMA、Runtime API 等权威来源，确认执行模型与内存模型的依据。 |
| “业界里成熟的 SGEMM 实现大概长什么样？” | [参考资料清单](/zh/references#示范仓库与生产级样例) | 用成熟仓库与官方样例对照本仓库的教学型 kernel 阶梯。 |
| “看完站内内容后下一步该学什么？” | [延伸阅读路线](/zh/resources/further-reading) | 把邻近主题整理成有顺序的学习路线，而不是把选择权全部丢给读者。 |
| “我该怎么把 benchmark 异常变成可验证的证据？” | [诊断闭环](/zh/methodology/diagnosis-loop) + [Profiler 与工具资料](/zh/references#profiler、工具与诊断资料) | 把站内诊断工作流与外部 profiler / 工具连接起来。 |

## 精选资源架

### 先查官方文档，统一术语与约束

当你需要确认白皮书里的某个说法是否有权威依据时，从这里开始。

- [CUDA C++ Programming Guide](/zh/references#cuda-与-nvidia-官方文档)：执行模型、同步语义、内存层级。
- [WMMA / Tensor Core 资料](/zh/references#cuda-与-nvidia-官方文档)：片段约束、对齐条件、混合精度行为。
- [CUDA Runtime API 与 Compute Sanitizer](/zh/references#profiler、工具与诊断资料)：更贴近调试与验证实践。

### 再看论文与性能模型，补上“为什么”

这些资源帮助你理解 kernel 阶梯背后的设计逻辑，而不只是记 API。

- [基础论文与性能心智模型](/zh/references#论文与性能心智模型) 解释为什么 blocking、数据复用、算术强度会主导 SGEMM。
- [延伸阅读：Roofline 思维](/zh/resources/further-reading#面向-sgemm-的-roofline-思维) 把这些抽象概念重新落回调优判断。 

### 想对照成熟实现时，再打开示范仓库

这些链接帮助你看清本仓库在哪些地方是为了教学而故意简化，哪些地方与生产实现共享核心思路。

- [CUTLASS、BLIS 与 CUDA Samples](/zh/references#示范仓库与生产级样例) 展示了不同风格的 GEMM 实现。
- [Kernel 阶梯](/zh/architecture/kernel-ladder) 依然重要，因为它能作为阅读工业级模板库前的解释性底图。

### 当性能数字不再自解释时，切到工具层

如果“GFLOPS 变了”已经不足以说明问题，就该用工具证据补上上下文。

- [Profiler、工具与诊断资料](/zh/references#profiler、工具与诊断资料) 收纳了 Nsight Compute、Nsight Systems、Occupancy Calculator 和正确性工具。
- [Benchmark 范围](/zh/validation/benchmark-scope) 与 [可复现性](/zh/validation/reproducibility) 解释这些工具结果应该如何被汇报与解释。

## 推荐学习路线

### 路线 1：先把内存讲清楚，再谈下一步优化

1. 先读 [Memory Flow](/zh/architecture/memory-flow)。
2. 用 [CUDA 内存速查表](/zh/cuda-memory-cheatsheet) 快速检查一个 warp 在加载什么、复用发生在哪、shared memory 改变了什么访问模式。
3. 如果还想把这些概念升维，继续读 [延伸阅读：GEMM 分块](/zh/resources/further-reading#gemm-分块与层级化思维)。

### 路线 2：别用口号解释 Tensor Core

1. 先读 [Tensor Core 路径](/zh/architecture/tensor-core-path) 和 [Tensor Core WMMA](/zh/kernel-tensor-core)。
2. 再打开 [WMMA 与混合精度资料](/zh/references#cuda-与-nvidia-官方文档)，核对 shape、对齐、fallback 条件。
3. 最后转到 [延伸阅读：Tensor Core 约束](/zh/resources/further-reading#tensor-core-约束与-fallback-设计)，把常被省略的前置条件补齐。

### 路线 3：从“benchmark 变了”走向 profiler 驱动诊断

1. 从 [诊断闭环](/zh/methodology/diagnosis-loop) 开始。
2. 配合 [Nsight 与 Occupancy 资料](/zh/references#profiler、工具与诊断资料) 使用。
3. 想把症状、指标、结论串成完整流程时，继续看 [延伸阅读：从症状到证据的 profiling](/zh/resources/further-reading#从症状到证据的-profiling-路线)。

## 站内相关页面

- [参考资料清单](/zh/references)
- [CUDA 内存速查表](/zh/cuda-memory-cheatsheet)
- [延伸阅读路线](/zh/resources/further-reading)
- [诊断闭环](/zh/methodology/diagnosis-loop)
- [验证概览](/zh/validation/)
