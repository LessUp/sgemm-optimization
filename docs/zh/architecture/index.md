---
title: 架构概述
---

# 架构概述

本节是 SGEMM 系统的规范化地图，记录每个组件的职责、数据如何流经内存层级、内核选择逻辑何时触发，以及系统不变量与边界在哪里。在相信任何 benchmark 结论或面试时的架构表述之前，请先阅读本节。

<ThemedFigure
  :wide="true"
  light="/figures/kernel-ladder-light.svg"
  dark="/figures/kernel-ladder-dark.svg"
  alt="展示 naive FP32、tiled FP32、bank-free FP32、double buffer、Tensor Core WMMA，并连接架构、验证与研究辅助轨的 kernel 阶梯图。"
  caption="阶梯是一张瓶颈迁移地图，不是奖杯陈列架。每一级存在的理由，是前一级暴露了新的上限。"
/>

## 技术论断

现代 NVIDIA GPU 上的 SGEMM 优化是一系列瓶颈类别的迁移序列。从朴素 FP32 到 Tensor Core WMMA，需要按顺序解决四个不同问题：DRAM 饱和、共享内存 bank 冲突、staging 与计算的重叠，以及 WMMA 硬件约束。本仓库把 kernel 实现结构化为每个阶段只暴露一个问题、保持前序阶段不变，使每一个架构决策的性能效果都可以独立观察。

## 组件清单

| 组件 | 层次 | 主要职责 |
|------|------|----------|
| `src/main.cu` | 驱动层 | 程序入口，负责把 CLI 解析和 benchmark 编排接起来 |
| `src/cli_parser.cuh` | 驱动支撑 | 把模式标志、shape 和运行参数解析成 `BenchmarkConfig` |
| `src/benchmark_runner.cuh` | 驱动支撑 | 用同一个二进制执行 benchmark 与 verification 流程 |
| `src/kernels/naive_sgemm.cuh` | Kernel | FP32 基线，承受完整全局内存加载代价 |
| `src/kernels/tiled_sgemm.cuh` | Kernel | 协作式 tile 加载入共享内存；SMEM 复用 |
| `src/kernels/bank_conflict_free_sgemm.cuh` | Kernel | Padding 消除共享内存 bank 冲突 |
| `src/kernels/double_buffer_sgemm.cuh` | Kernel | 将下一 tile 的 staging 与当前计算重叠 |
| `src/kernels/tensor_core_sgemm.cuh` | Kernel | 在硬件对齐的 tile 上执行 WMMA fragment 累加 |
| `src/utils/cuda_utils.cuh` | 工具 | CUDA 错误宏、RAII 设备内存封装与设备信息 |
| `src/utils/verify.cuh` | 工具 | 基于 cuBLAS 的正确性校验与容差策略 |
| `tests/test_sgemm.cu` | 测试 | 基于 GPU 和 cuBLAS 参考的正确性测试 |

## 内存层级数据流

每一步 kernel 优化都对应一次内存访问代价来源的变化：

```
Naïve:        [全局内存]  → 寄存器              (DRAM 受限)
Tiled:        [全局内存]  → SMEM → 寄存器       (SMEM 复用，暴露 bank 冲突)
Bank-Free:    [全局内存]  → SMEM+pad → 寄存器   (无冲突，暴露延迟)
Dbl-Buffer:   [全局内存]  → 双 SMEM → 寄存器    (staging 隐藏在计算后面)
Tensor Core:  [全局内存]  → SMEM → WMMA frags   (硬件加速矩阵累加)
```

Memory Flow 页面将以具体的地址、stride、tile 维度和加载模式呈现这一数据流。

## 设计不变量

以下属性在所有 kernel 阶段保持不变，是架构正确性契约的一部分：

1. **全程行主序布局。** 所有矩阵 A、B、C 使用行主序存储，任何 kernel 均不隐式假定列主序。
2. **向量化加载以 float4 为粒度。** 受益于更宽加载的 kernel 使用 `float4` 来最大化每条指令的内存带宽。
3. **约束不满足时回退。** Tensor Core 入口路径在 shape guard 不满足时回退到 FP32 路径，benchmark 数据绝不来自回退激活的运行。
4. **以 epsilon 边界验证正确性。** 测试框架以每元素容差 `1e-3` 与 cuBLAS 参考比对，kernel 正确性通过测量而非假定来保证。
5. **计时在 CUDA graph 边界之外。** Benchmark 计时包含完整的设备调用与同步，冷启动和预热行为在每份结果中均有说明。

## Kernel 选择与回退逻辑

`src/main.cu` 中的入口路径根据设备能力查询和矩阵维度检查来选择 kernel 层次：

| 条件 | 选择路径 |
|------|----------|
| 任意 GPU，任意尺寸 | FP32 阶梯（朴素 → 双缓冲） |
| SM ≥ 7.0，尺寸可被 WMMA tile 整除 | Tensor Core WMMA 路径 |
| SM ≥ 7.0，尺寸不对齐 WMMA | FP32 路径（回退） |
| SM < 7.0 | FP32 路径（回退） |

纯 benchmark 直接在预先验证的 shape 上调用 Tensor Core kernel，安全入口路径使用运行时 guard。

## 架构决策

### 1. 阶梯，而非技巧清单

每个 kernel 解决一类瓶颈：

1. **Naïve** 建立算术强度下界，暴露 DRAM 饱和。
2. **Tiled** 协作式地将数据加载到共享内存，暴露 bank 冲突作为新上限。
3. **Bank-Free** 通过 padding 消除冲突，暴露 staging 延迟。
4. **Double Buffer** 将下一 tile 的 staging 与当前计算重叠，减少停顿周期。
5. **Tensor Core** 在严格的对齐和设备约束下使用硬件融合矩阵累加。

关键在于每一级都有单一存在理由和单一可测量的架构效果。

### 2. 验证作为架构一等公民

项目刻意区分两种环境各自能证明什么：

| 主张 | 本地 CUDA GPU | 托管 CI |
|------|---------------|---------|
| 编译成功 | ✓ | ✗ |
| 输出正确性 vs. cuBLAS | ✓ | ✗ |
| Benchmark 性能结论 | ✓ | ✗ |
| 仓库结构与文档 | ✓ | ✓ |
| VitePress Pages 可构建性 | ✓ | ✓ |

这不只是流程卫生问题，它影响读者能从 CI 绿灯中信任哪些主张。

### 3. Tensor Core 作为显式快路径

FP32 阶梯与 Tensor Core 路径是两条独立的层次。仓库同时暴露二者：

- WMMA 的 benchmark 主张只在对齐 shape 且 SM ≥ 7.0 设备上提出。
- FP32 阶梯是完整的独立教学路径，不依赖 Tensor Core 硬件。
- 回退行为通过测试验证，而非假定。

## 系统地图与阅读路径

| 需求 | 去往 |
|------|------|
| 完整组件和数据流图 | [系统蓝图](./system-blueprint) |
| 逐级 kernel 的瓶颈迁移解释 | [Kernel 阶梯](./kernel-ladder) |
| 内存层级与加载模式分析 | [Memory Flow](./memory-flow) |
| WMMA 选择、shape guard、回退 | [Tensor Core 路径](./tensor-core-path) |
| 有序教学路径、面试框架 | [学院](../academy/) |
| 正确性策略与 benchmark 范围 | [验证](../validation/) |
| 外部参考文献与对比 | [研究](../research/) |

## 评审者快速路径

1. 本页：架构论断与不变量。
2. [系统蓝图](./system-blueprint)：含数据流的完整组件清单。
3. [验证概览](../validation/)：证据能证明什么，不能证明什么。
4. [Benchmark 结果](../validation/benchmark-results)：附带范围说明的数据。
5. [学院](../academy/)：用于面试答辩的有序解释。
