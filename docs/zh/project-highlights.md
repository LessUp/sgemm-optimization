---
title: 项目亮点
---

# 项目亮点

本页现在只保留快速导览功能。完整的“为什么会这样设计”叙事已经迁移到[架构章节](/zh/architecture/)。

## 这个仓库真正想证明什么

- SGEMM 优化可以被讲成一条推理链，而不是技巧清单。
- 只有把正确性策略和 benchmark 范围绑定起来，性能结论才可信。
- 只有明确约束与 fallback 行为，Tensor Core 加速才有解释价值。

## 不同问题该从哪一页开始

| 如果你想知道…… | 建议先看 |
|----------------|-----------|
| 为什么整个系统这样组织 | [架构概述](/zh/architecture/) |
| 为什么 kernel 要按这个顺序出现 | [Kernel 阶梯](/zh/architecture/kernel-ladder) |
| 数据如何在内存层级中流动 | [Memory Flow](/zh/architecture/memory-flow) |
| WMMA 什么时候能用、什么时候会被拒绝 | [Tensor Core 路径](/zh/architecture/tensor-core-path) |
| 面试时该如何讲这个项目 | [面试手册](/zh/interview-playbook) |

## 给评审者的快速路径

1. [架构概述](/zh/architecture/)
2. [Kernel 阶梯](/zh/architecture/kernel-ladder)
3. [Benchmark 结果](/zh/benchmark-results)
4. [面试手册](/zh/interview-playbook)
5. [参考文献](/zh/references)
