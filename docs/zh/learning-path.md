---
title: 学习路径
---

# 学习路径

按仓库设计的教学顺序跟随优化阶梯



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



## 开始之前

- 确保环境符合 [快速上手](/zh/getting-started)
- 如需先了解仓库级地图，查看 [架构概览](/zh/architecture)
- 如需规范需求，参考 [规范索引](https://github.com/LessUp/sgemm-optimization/tree/master/openspec/specs/)
