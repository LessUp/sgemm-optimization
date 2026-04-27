---
layout: default
title: 学习路径
nav_order: 3
permalink: /zh/docs/learning-path
lang: zh-CN
page_key: zh-learning-path
lang_ref: learning-path
---

# 学习路径
{: .fs-8 }

按仓库设计的教学顺序跟随优化阶梯
{: .fs-6 .fw-300 }

---

## 推荐顺序

| 步骤 | Kernel | 为什么在这里 |
|------|--------|-------------|
| 1 | [Naive](kernel-naive) | 建立基线代价模型 |
| 2 | [Tiled](kernel-tiled) | 引入共享内存复用 |
| 3 | [Bank-Free](kernel-bank-free) | 展示共享内存布局为何重要 |
| 4 | [Double Buffer](kernel-double-buffer) | 加入分阶段和重叠概念 |
| 5 | [Tensor Core](kernel-tensor-core) | 进入 WMMA 和混合精度硬件 |

---

## 每个阶段教什么

### Naive → Tiled

- 线程/块映射
- 内存合并访问
- 共享内存复用

### Tiled → Bank-Free

- 32 路 bank 共享内存行为
- 为什么 `[32][33]` 很重要

### Bank-Free → Double Buffer

- 流水线思维
- Tile 分阶段加载与延迟隐藏

### Double Buffer → Tensor Core

- WMMA 片段
- 混合精度
- 不支持形状的安全回退行为

---

## 建议阅读节奏

1. 先编译运行项目
2. 阅读某一阶段的 kernel 页面
3. 再次运行 benchmark
4. 对比当前阶段与上一阶段的代码
5. 理解当前优化后再进入下一阶段

---

## 开始之前

- 确保环境符合 [快速上手](getting-started)
- 如需先了解仓库级地图，查看 [架构概览](architecture)
- 如需规范需求，参考 [规范索引](../../specs)
