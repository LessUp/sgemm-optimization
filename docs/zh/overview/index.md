---
title: 项目导读
---

# 项目导读

这是整套白皮书站点的执行摘要。想先弄清楚项目定位、读者地图和推荐阅读顺序时，请先从这里进入。

## 这个项目是什么

本仓库是一个围绕分阶段 kernel 阶梯组织的 CUDA SGEMM 研究：

1. 可读的 FP32 基线
2. shared memory 分块
3. bank conflict 缓解
4. double buffer 重叠
5. 带保护的 Tensor Core WMMA

目标不是和生产级 GEMM 库正面对打，而是展示一条优化论证如何被建立、被捍卫，以及被限制。

## 这套站点面向谁

| 读者 | 先看哪里 | 为什么 |
|---|---|---|
| 面试官 | [架构](../architecture/) | 最快看到系统清晰度和项目差异化 |
| 准备项目讲解的候选人 | [学院](../academy/) | 提供教学顺序和面试友好的表达框架 |
| CUDA 学习者 | [架构](../architecture/) 然后 [学院](../academy/) | 先建立概念阶梯，再进入代码细节 |
| 性能怀疑者 | [验证](../validation/) | 明确证据从哪里开始，到哪里结束 |
| 偏研究型读者 | [研究](../research/) | 提供论文、对照仓库与技术谱系 |

## 站点结构

| 章节 | 核心职责 |
|---|---|
| [导读](./) | 定位说明与阅读策略 |
| [架构](../architecture/) | 系统地图、瓶颈、关键约束 |
| [学院](../academy/) | 按顺序学习优化阶梯 |
| [验证](../validation/) | 正确性与 benchmark 的信任边界 |
| [研究](../research/) | 参考资料、相关工作与演进思考 |

## 快速阅读路线

### 评审路径

1. [架构概述](../architecture/)
2. [Kernel 阶梯](../architecture/kernel-ladder)
3. [验证概览](../validation/)
4. [相关开源项目](../research/related-projects)

### 候选人路径

1. [学院导览](../academy/)
2. [学习路径](../academy/learning-path)
3. [诊断闭环](../academy/diagnosis-loop)
4. [演进思考](../research/evolution)

### 构建者路径

1. [快速上手](./getting-started)
2. [正确性策略](../validation/correctness-policy)
3. [Benchmark 范围](../validation/benchmark-scope)
4. [参考资料清单](../research/references)
