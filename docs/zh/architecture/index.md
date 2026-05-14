---
title: 架构概述
---

# 架构概述

本节是 SGEMM 系统的规范化地图：解释为什么要这样设计、数据如何流动、每一级 kernel 为什么出现，以及 Tensor Core 何时可以接管计算。

## 为什么会有这套设计

这个仓库不是“放一个快 kernel 再贴一张跑分图”。它被组织成一条工程推理链：

- 从可读的 FP32 基线开始
- 先暴露瓶颈，而不是掩盖瓶颈
- 每次只引入一个新的架构思想
- 始终把正确性与 benchmark 范围讲清楚
- 当 Tensor Core 条件不满足时，保留安全路径

这样的结构让项目同时适合学习、评审和面试：读者可以先解释 **为什么** 某个 kernel 存在，再讨论它快了多少。

## 系统地图

| 层次 | 责任 | 下一步阅读 |
|------|------|-----------|
| Kernel 阶梯 | 解释从朴素 FP32 到 WMMA 的优化链路 | [Kernel 阶梯](/zh/architecture/kernel-ladder) |
| Memory Flow | 用一个系统视角解释全局内存访问、共享内存复用、bank 冲突与双缓冲 | [Memory Flow](/zh/architecture/memory-flow) |
| Tensor Core 路径 | 解释 WMMA 选择逻辑、FP32→FP16 staging、shape guard 与 fallback 行为 | [Tensor Core 路径](/zh/architecture/tensor-core-path) |
| Kernel 深入页 | 单独解释每个 kernel 的实现细节 | [朴素](/zh/kernel-naive)、[分块](/zh/kernel-tiled)、[消除 Bank Conflict](/zh/kernel-bank-free)、[双缓冲](/zh/kernel-double-buffer)、[Tensor Core WMMA](/zh/kernel-tensor-core) |

## 影响仓库结构的架构决策

### 1. 优化被呈现为阶梯，而不是技巧清单

每一级 kernel 都对应一个明确的瓶颈类别：

1. **Naïve** 建立代价模型并暴露复用不足。
2. **Tiled** 用共享内存复用换取更少的全局内存压力。
3. **Bank-Free** 通过 padding 消除可避免的共享内存冲突。
4. **Double Buffer** 通过重叠 staging 和 compute 隐藏部分内存延迟。
5. **Tensor Core** 在显式设备与 shape 条件下提高吞吐上限。

目标不是“后面的 kernel 在所有 GPU 上都一定更快”，而是每一步都要有清楚的存在理由和可解释的架构效果。

### 2. 数据移动是系统主线

本仓库把 SGEMM 性能问题主要描述为“数据在哪里、什么时候移动”：

- 从全局内存进入 SM
- 从全局内存进入共享 tile
- 从共享内存进入寄存器或 WMMA fragment
- 从分阶段 tile 回写到输出矩阵 C

这就是为什么架构章节把 memory flow 作为一级主题，而不是把相关解释散落到各 kernel 页面里。

### 3. Tensor Core 是可选快路径，不是唯一道路

仓库同时暴露两种路径：

- **安全 FP32 入口**：必要时执行类型转换，并在 WMMA 不可用时回退
- **纯 compute-only WMMA 路径**：只在友好 shape 下用于测量原始 Tensor Core 行为

这样 benchmark 才能保持诚实：不支持的维度不会被悄悄统计成 Tensor Core 成绩。

### 4. 验证边界也是架构的一部分

项目刻意区分不同环境下可以信任什么：

| 范围 | 本地 CUDA GPU | 托管 CI |
|------|---------------|---------|
| CUDA 编译 | 是 | 否 |
| 运行时正确性 | 是 | 否 |
| Benchmark 性能 | 是 | 否 |
| 文档、OpenSpec 与仓库完整性 | 是 | 是 |
| Pages 可构建性 | 可选 | 是 |

这不仅是流程说明，也会直接影响架构叙事：性能结论必须回到正确的运行环境里解释。

## 推荐阅读顺序

1. 先在本页建立系统地图。
2. 阅读 [Kernel 阶梯](/zh/architecture/kernel-ladder) 理解优化链路。
3. 阅读 [Memory Flow](/zh/architecture/memory-flow) 理解这条链路背后的数据移动逻辑。
4. 在解释 WMMA benchmark 前，先阅读 [Tensor Core 路径](/zh/architecture/tensor-core-path)。
5. 当你需要实现细节而不是系统原因时，再跳转到各个 kernel 深入页。

## 相关资源

- [资源中心](/zh/resources/)
- [验证概览](/zh/validation/)
- [学习路径](/zh/learning-path)
- [快速上手](/zh/getting-started)
- [稳定架构规范](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/specs/architecture/spec.md)
