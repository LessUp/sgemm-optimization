---
layout: default
title: CUDA 内存速查表
parent: 首页
nav_order: 11
permalink: /zh/docs/cuda-memory-cheatsheet/
lang: zh-CN
page_key: zh-cuda-memory-cheatsheet
lang_ref: cuda-memory-cheatsheet
---

# CUDA 内存速查表
{: .fs-8 }

一页读懂 SGEMM kernel 的关键内存知识
{: .fs-6 .fw-300 }

---

## 一张表看内存层级

| 层级 | 典型作用域 | 本仓库 SGEMM 用法 | 常见坑 |
|------|------------|------------------|--------|
| 全局内存 | 设备级 | 输入输出矩阵与中间缓冲区 | 访问不合并 |
| 共享内存 | block 级 | A/B 子矩阵 tile 复用 | 布局导致 bank 冲突 |
| 寄存器 | 线程级 | 累加器与片段中间值 | 寄存器压力拉低占用率 |
| L2 缓存 | 设备级缓存 | 帮助重复全局读 | 误以为缓存总能掩盖索引问题 |

---

## 合并访问速记

- 同一个 warp 的相邻线程，应尽量访问相邻地址。
- 当 `N` 很大时，`B[k * N + col]` 容易让相邻线程产生大步长访问。
- 分块不仅是复用数据，也是在重塑访问模式，让加载更合并。

```mermaid
flowchart LR
    A[全局内存加载 tile] --> B[共享内存缓存 tile]
    B --> C[寄存器累加]
    C --> D[写回全局内存]
```

---

## 共享内存 bank 冲突提示

| 访问模式 | 风险等级 | 常见修复 |
|----------|----------|----------|
| `tile[ty][tx]` 且 warp 访问连续 | 低 | 保持线程到列方向的连续映射 |
| 类似转置 `tile[tx][ty]` 且无 padding | 高 | 增加 padding，例如 `[32][33]` |
| 多阶段复用相同步长地址 | 中 | 复查分阶段布局与 lane 映射 |

bank 冲突不一定总是主瓶颈，但常常容易被忽略，也容易被过度归因。

---

## Tensor Core 内存提示

| 主题 | 记住这点 |
|------|----------|
| 对齐约束 | WMMA 路径通常要求维度按 16 对齐，片段处理才高效 |
| 数据转换 | 端到端耗时包含转换和 wrapper 逻辑 |
| 安全行为 | 不友好 shape 应回退到 FP32 路径 |
| 结果汇报 | 要区分端到端与仅计算数据 |

---

## 从 profiler 到动作

| 指标趋势 | 含义判断 | 下一步动作 |
|----------|----------|------------|
| `dram` 高、`sm` 低 | 可能以内存为主限制 | 优先改善合并访问与复用 |
| 高寄存器使用且占用率低 | 活跃 warp 被寄存器限制 | 减少线程私有临时状态 |
| 共享 bank 冲突尖峰 | 共享内存布局不匹配 | 引入 padding 或重排 lane |

---

## 读 kernel 的快速清单

1. 能否解释一个 warp 的全局内存访问顺序？
2. 共享内存布局是否显式考虑 bank 冲突？
3. 寄存器累加器是否受控且必要？
4. Tensor Core 回退行为是否清晰？
5. benchmark 标签是否和真实测量路径一致？

---

## 相关页面

- [Kernel 1: Naïve](kernel-naive/)
- [Kernel 3: Bank Conflict Free](kernel-bank-free/)
- [Kernel 5: Tensor Core](kernel-tensor-core/)
- [优化实战手册](optimization-playbook/)
