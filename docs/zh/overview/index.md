---
title: 项目导读
---

# 项目导读

这是 SGEMM 白皮书站点的定向页面。在进入更深层的章节之前，先用它了解项目的定位、目标读者和推荐阅读顺序。

## 这个项目是什么

本仓库是一个围绕五级 kernel 优化阶梯组织的 CUDA SGEMM 研究：

1. **朴素 FP32** — 基线代价模型，无共享内存复用
2. **Tiled FP32** — 共享内存 staging，算术强度随 tile 大小增长
3. **Bank-Free FP32** — padding 消除可避免的 bank 冲突
4. **Double Buffer** — 重叠 staging 与 compute，隐藏内存延迟
5. **Tensor Core WMMA** — 硬件 fragment 累加，受设备能力和 shape 约束保护

目标不是写出最快的 SGEMM 实现，而是展示一条优化论证如何被构建、约束和辩护——以一种在面试压力下可读、能被有经验的 CUDA 工程师审计的形式。

## 这套站点面向谁

| 读者 | 最佳首页 | 时间 |
|---|---|---|
| 审查系统清晰度的面试官 | [架构概述](../architecture/) | 8 分钟 |
| 准备讲解项目的候选人 | [学院导览](../academy/) | 5 分钟后跟随[学习路径](../academy/learning-path) |
| 从零开始学习 CUDA 的工程师 | 本页，然后[架构](../architecture/)，再[学院](../academy/) | 自定步调 |
| 关注证据质量的怀疑论者 | [验证概览](../validation/) | 12 分钟 |
| 追溯技术谱系的研究者 | [研究总览](../research/) | 自定步调 |

完整的分深度导航索引参见[阅读地图](./reader-map)。

## 站点结构

每个章节只做一件事。这是刻意为之：承担两个职责的页面，哪个都做不好。

| 章节 | 主要职责 | 它不是 |
|---|---|---|
| [导读](./) | 定向与阅读策略 | 不是架构章节的替代 |
| [架构](../architecture/) | 系统地图、瓶颈、约束条件 | 不是代码走读 |
| [学院](../academy/) | 有序的优化阶梯学习 | 不是参考手册 |
| [验证](../validation/) | 正确性与 benchmark 信任边界 | 不是性能声明 |
| [研究](../research/) | 参考资料、相关工作、演进思考 | 不是扩展参考书目 |

## 快速阅读计划

### 评审者路径（20 分钟）

1. [架构概述](../architecture/)
2. [Kernel 阶梯](../architecture/kernel-ladder)
3. [验证概览](../validation/)
4. [相关项目](../research/related-projects)

### 候选人路径（30 分钟）

1. [学院导览](../academy/)
2. [学习路径](../academy/learning-path)
3. [诊断闭环](../academy/diagnosis-loop)
4. [演进思考](../research/evolution)

### 构建者路径（自定步调）

1. [快速上手](./getting-started)
2. [系统蓝图](../architecture/system-blueprint)
3. [正确性策略](../validation/correctness-policy)
4. [Benchmark 范围](../validation/benchmark-scope)
5. [参考资料清单](../research/references)

## 为什么这套站点是白皮书，而不是项目秀场

大多数项目文档描述构建了什么。这套站点论证了每个架构决策 *为何* 存在，*什么证据* 约束了论点，以及 *推理在哪里停止*。

当面试官问："bank-free kernel 为什么存在？它实际改善了什么？"——这种区别就变得重要。展示型回答是："它更快。"白皮书型回答是："多个线程映射到同一个 bank 时，共享内存 bank 冲突会串行化访问。在 tile 布局上 padding 一个元素，将每一列移到不同的 bank，消除多路冲突。这种改善在容易产生冲突的 shape 上是真实的、可测量的；它不是普遍的。"

这就是本站点在所有五个 kernel 阶段、在架构和验证两个维度上所追求的表达精度。
