---
title: Memory Flow
---

# Memory Flow

本页把 SGEMM 看成一个数据移动系统：不同 kernel 的主要区别，不是公式本身，而是它们如何让矩阵 tile 在全局内存、共享内存、寄存器和专用计算单元之间流动。

## 端到端数据路径

```text
全局内存中的 A、B
    ↓
block 级 coalesced 加载
    ↓
共享内存 staging
    ↓
寄存器或 WMMA fragment 消费
    ↓
FP32 累加
    ↓
回写到全局内存中的 C
```

架构目标就是让这条路径的每一步更便宜、更可复用，或者调度得更合理。

## 按阶段看的系统视角

| 阶段 | 数据流发生了什么变化 | 为什么重要 |
|------|----------------------|-----------|
| Naïve | 每个乘加步骤都直接从全局内存取数 | 复用极少，B 的局部性很差 |
| Tiled | block 先把 A/B tile 搬进共享内存 | 一次全局加载可以被很多次计算摊销 |
| Bank-Free | 对共享 tile 做 padding，改变 bank 映射 | 共享内存带宽更稳定、更可预测 |
| Double Buffer | 两个共享内存槽在加载与计算之间交替 | 一部分加载延迟可以被计算隐藏 |
| Tensor Core | FP16 tile 喂给 WMMA fragment，再以 FP32 累加 | 吞吐更高，但 staging 规则更严格 |

## 1. 全局内存行为：第一层问题

在朴素 kernel 中，每个输出元素都依赖直接的全局内存读取，这会暴露两个昂贵现实：

- **A** 的一行会被反复访问
- **B** 的一列会形成 stride 较大的访问模式

因此，第一步架构动作不是“多做计算”，而是“少做浪费的数据移动”。

## 2. 共享内存 staging：第一次大改造

Tiling 在每个 thread block 内引入了一个可复用工作集。

### 发生了什么

- block 协作加载 **A** 的一个 tile
- block 协作加载 **B** 的一个 tile
- 所有线程先重复利用这两个 tile，再加载下一组数据

### 新责任是什么

共享内存虽然快，但 block 必须尊重它的协作规则：

- 所有加载完成后才能开始计算
- 所有计算结束后才能覆盖旧 buffer
- tile 大小会影响 occupancy 和共享内存占用

所以 tiled kernel 中的 `__syncthreads()` 是正确性边界，不只是性能细节。

## 3. Bank 冲突：进入共享内存之后仍然可能出问题

把数据搬进共享内存，并不意味着访问天然高效。如果很多线程同时撞到同一个 bank，访问仍然会串行化。

### 仓库策略

Bank-free kernel 把共享内存领先维从 `TILE_SIZE` 改成 `TILE_SIZE + 1`。

### 系统效果

- 逻辑算法不变
- 物理地址步长改变
- 常见冲突模式被打散

这是一个内存系统决策，而不是数值决策。它的目标是让 staged 数据更容易被 SM 以 warp 速度服务。

## 4. Buffer 调度：从“存哪里”到“什么时候存”

双缓冲把 memory flow 从单一 staging 槽位，改成一个流水线。

```text
第 t 轮：   在 buffer 0 上计算   | 预加载 buffer 1
第 t+1 轮： 在 buffer 1 上计算   | 预加载 buffer 0
```

这一步的价值不在于新增复用，而在于重叠：

- 当前数据仍然可供计算
- 下一份数据可以提前移动
- tile 循环开始更像 producer/consumer 调度

所以双缓冲即便没有朴素→分块那样大的提升，也仍然属于一级架构话题。

## 5. Tensor Core staging：更快单元意味着更严的流动规则

Tensor Core 路径又增加了一层 staging 规则：

- 输入需要准备成 FP16
- 维度必须满足 WMMA tile 对齐条件
- 计算消费的是 warp 级 fragment，而不是标量寄存器循环

输出仍然以 FP32 累加，因此本仓库把这一路径描述为“混合精度”，而不是纯 FP16。

关于 guard 与 fallback 的决策逻辑，参见 [Tensor Core 路径](/zh/architecture/tensor-core-path)。

## Memory Flow 设计原则

### 先保证 coalescing，再谈花活

架构首先优先让全局内存访问模式合理，再加入更激进的优化。

### 先建立复用，再追求峰值吞吐

共享内存分块出现在 Tensor Core 之前，因为清楚的复用故事比硬件特化快路径更容易验证，也更容易讲清楚。

### 先追求可解释，再追求英雄数字

padding 消除 bank 冲突、以及显式 fallback 逻辑，本质上体现的是同一个价值：仓库更重视可解释行为，而不是脆弱的“只在最佳情况成立”的结论。

## 深入链接

- [分块内核](/zh/kernel-tiled)
- [消除 Bank Conflict](/zh/kernel-bank-free)
- [双缓冲](/zh/kernel-double-buffer)
- [Tensor Core WMMA](/zh/kernel-tensor-core)
- [CUDA 内存速查表](/zh/cuda-memory-cheatsheet)
