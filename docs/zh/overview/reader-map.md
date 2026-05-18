---
title: 阅读地图
---

# 阅读地图

本页是 SGEMM 白皮书站点的导航索引。根据你的目的、背景和可用时间选择合适的入口。

## 按目的选择入口

| 你是谁 | 最佳入口 | 预计时间 |
|---|---|---|
| 评审系统设计清晰度的面试官 | [架构概述](../architecture/) | 8 分钟 |
| 准备讲解项目的候选人 | [学院导览](../academy/) + [学习路径](../academy/learning-path) | 25 分钟 |
| 从零开始学习 CUDA 的工程师 | [快速上手](./getting-started) → [架构](../architecture/) → [学院](../academy/) | 45 分钟 |
| 关心证据质量的怀疑论者 | [验证概览](../validation/) → [Benchmark 结果](../validation/benchmark-results) | 12 分钟 |
| 追溯技术谱系的研究者 | [研究总览](../research/) → [论文索引](../research/papers) → [相关项目](../research/related-projects) | 20 分钟 |
| 想要最快速概览的读者 | 本页 + [架构概述](../architecture/) | 5 分钟 |

## 白皮书论证结构

站点被组织为一系列有据可查的主张：

```
主张：SGEMM 优化应该被讲成一条推理链
  └─ 架构  → 系统地图、约束条件、瓶颈阶梯
  └─ 学院  → 带有因果解释的有序 kernel 学习
  └─ 验证  → 正确性策略、benchmark 范围、信任边界
  └─ 研究  → 论文谱系、相关仓库、演进思考
```

每个章节只做一件事。阅读地图帮助你直接跳到匹配你问题的章节。

## 阅读深度分级

### 第一级：5 分钟评审快速扫描

1. [架构概述](../architecture/) — 系统主张与 kernel 阶梯
2. [验证概览](../validation/) — 证据的能力边界

### 第二级：20 分钟技术审查

1. [架构概述](../architecture/)
2. [Kernel 阶梯](../architecture/kernel-ladder)
3. [Memory Flow](../architecture/memory-flow)
4. [验证概览](../validation/)
5. [Benchmark 范围](../validation/benchmark-scope)

### 第三级：完整白皮书阅读

遵循学院中的[学习路径](../academy/learning-path)，该页给出了从入门定向到每个 kernel 深入解析的完整阅读顺序。

## 跨章节依赖关系

部分页面存在先后顺序：

- 先读 **Memory Flow**，再读 **Tensor Core 路径**。
- 先读 **Benchmark 范围**和**可复现性**，再读 **Benchmark 结果**。
- 先读 **验证概览**，再读 **性能案例库**。
- 先通过**学习路径**建立阶梯概念，再打开各 kernel 深入页面。

## 相关页面

- [快速上手](./getting-started) — 环境搭建与第一次运行
- [系统蓝图](../architecture/system-blueprint) — 完整组件依赖图
- [性能模型](../validation/performance-model) — 阶梯背后的量化代价模型
