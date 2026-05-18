---
title: 学院
---

# 学院

学院是本仓库的有序学习界面。架构提供系统地图，学院提供教学顺序——每个 kernel 阶段以什么顺序讲解，以及为什么这个顺序不可颠倒。

## 组织原则

把 kernel 理解为一连串瓶颈转移，而不是技巧清单：

| 阶段 | 暴露的瓶颈 | 引入的结构变化 |
|---|---|---|
| 朴素 FP32 | 无限制的 DRAM 流量 | 建立代价模型 |
| Tiled FP32 | 冗余全局内存读取 | 共享内存 staging |
| Bank-Free FP32 | 共享内存 bank 冲突 | Tile padding |
| Double Buffer | 关键路径上的内存延迟 | 重叠 staging 与 compute |
| Tensor Core WMMA | FP32 吞吐上限 | 硬件 fragment 累加 |

每个后续页面都假设前一页已经解释了为什么引入额外复杂性是合理的。乱序阅读会让因果链变得不可见。

## 学院地图

| 学习轨道 | 目的 | 从这里开始 |
|---|---|---|
| 定向 | 在打开任何 kernel 页面之前，先了解穿越阶梯的路线 | [学习路径](./learning-path) |
| 实验纪律 | 避免从草率的测量中得出结论 | [Benchmark 纪律](./benchmark-discipline) |
| 瓶颈推理 | 把症状转化为下一个可辩护的架构改变 | [诊断闭环](./diagnosis-loop) |
| Kernel 深入解析 | 按顺序检视实际的优化阶段 | [朴素 Kernel](./kernel-naive) |
| 记忆辅助 | 快速复习内存层次结构和调优启发式规则 | [CUDA 内存速查表](./cuda-memory-cheatsheet) |

## 推荐阅读顺序

1. [学习路径](./learning-path) — 任何 kernel 之前的定向
2. [朴素 Kernel](./kernel-naive) — 代价模型基线
3. [分块 Kernel](./kernel-tiled) — 共享内存复用
4. [消除 Bank Conflict](./kernel-bank-free) — 冲突 shape 下的稳定性
5. [双缓冲](./kernel-double-buffer) — 延迟隐藏
6. [Tensor Core WMMA](./kernel-tensor-core) — 受保护的吞吐上限
7. [诊断闭环](./diagnosis-loop) — 把测量结果转化为决策
8. [优化作战手册](./optimization-playbook) — 结构化调优流程

## 面试答辩框架

在评审中为任何 kernel 阶段辩护时，使用这四步结构：

1. **说明当前瓶颈** — 什么资源被饱和或被浪费？
2. **说明具体结构变化** — 这个 kernel 在硬件层面做了什么不同的事？
3. **说明证据要求** — 什么测量结果能确认这个改变有效？
4. **说明约束条件** — 什么假设或 shape 条件限制了这个改善？

这个顺序把讨论维持在工程推理层面，而不是 benchmark 截图层面。学院的设计目标就是让你能够为五个阶段中的每一个提供可辩护的答案。

## 学院不是什么

学院不是 CUDA 编程参考手册。参考资料请使用 [CUDA C++ 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)和本节中的 [CUDA 内存速查表](./cuda-memory-cheatsheet)。

学院不是源代码阅读的替代品。每个 kernel 页面解释架构推理，代码本身包含实现。两者都是对任何阶段给出完整说明的必要条件。

## 相关资源

- [架构概述](../architecture/) — 将阶梯置于背景中的系统地图
- [验证概览](../validation/) — 学院学习过程中产生的任何数字的信任边界
- [性能模型](../validation/performance-model) — 每个阶梯阶段背后的分析代价模型
