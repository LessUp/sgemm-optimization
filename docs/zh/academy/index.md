---
title: 学院
---

# 学院

学院是本仓库的顺序化学习表面。架构负责给地图，学院负责给教学顺序。

## 本节的基本规则

把 kernel 读成一条瓶颈转移链：

1. 先建立代价模型
2. 再改变数据复用
3. 再稳定 shared memory 行为
4. 再重叠加载与计算
5. 最后引入受保护的混合精度

这个顺序很重要，因为后面的页面都默认前面的页面已经解释清楚，为什么新增复杂度是合理的。

## 学院地图

| 轨道 | 目的 | 从哪里开始 |
|---|---|---|
| 导航 | 先建立学习顺序 | [学习路径](./learning-path) |
| 实验纪律 | 避免从糟糕测量里得出结论 | [Benchmark 纪律](./benchmark-discipline) |
| 瓶颈推理 | 把症状变成下一个可辩护的改动 | [诊断闭环](./diagnosis-loop) |
| Kernel 深入 | 进入各阶段实现细节 | [朴素内核](./kernel-naive) |
| 速查工具 | 快速复习内存与调优要点 | [CUDA 内存速查表](./cuda-memory-cheatsheet) |

## 推荐顺序

1. [学习路径](./learning-path)
2. [朴素内核](./kernel-naive)
3. [分块内核](./kernel-tiled)
4. [消除 Bank Conflict](./kernel-bank-free)
5. [双缓冲](./kernel-double-buffer)
6. [Tensor Core WMMA](./kernel-tensor-core)
7. [诊断闭环](./diagnosis-loop)
8. [优化作战手册](./optimization-playbook)

## 面试里的表达骨架

当你需要快速讲清楚这个项目时，请按这个顺序组织：

1. 先说当前瓶颈是什么
2. 再说结构上改了什么
3. 再说什么证据能证明这个改动有效
4. 最后说设计还受什么约束

这个顺序能让讨论保持技术含量，也能避免掉进“反正更快了”的空洞表达。
