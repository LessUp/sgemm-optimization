---
title: Kernel 阶梯
---

# Kernel 阶梯

Kernel 阶梯是这个仓库的主推理链。每一级都因为前一级暴露了某个瓶颈，而下一步正好针对它做出回应。

## 阶梯总览

| 阶段 | 暴露的主要瓶颈 | 架构动作 | 代价/约束 | 深入页 |
|------|----------------|----------|-----------|--------|
| Naïve | 全局内存重复访问、复用差 | 一个线程负责一个输出元素 | 可读，但明显受内存限制 | [朴素内核](/zh/kernel-naive) |
| Tiled | 全局内存带宽压力 | 用共享内存分块并在 block 内复用 | 需要 barrier 与 tile-size 权衡 | [分块内核](/zh/kernel-tiled) |
| Bank-Free | 共享内存争用 | 通过 padding 消除 bank 冲突模式 | 共享内存占用略增 | [消除 Bank Conflict](/zh/kernel-bank-free) |
| Double Buffer | 加载与计算串行化 | 用 ping-pong 缓冲区重叠加载与计算 | 共享内存更多，调度更复杂 | [双缓冲](/zh/kernel-double-buffer) |
| Tensor Core | CUDA Core 吞吐上限 | 在 Tensor Core 上执行 warp 级 WMMA | 需要 FP16 staging、shape guard 和 fallback 策略 | [Tensor Core WMMA](/zh/kernel-tensor-core) |

## 为什么要用“阶梯”来讲

很多 SGEMM demo 把多个 kernel 当作彼此独立的小技巧。本仓库把它们放进同一条叙事链里，让读者在每一步都能回答四个问题：

1. **看到了什么瓶颈？**
2. **用了什么架构机制去处理？**
3. **引入了什么新的代价或约束？**
4. **这一步应该如何验证？**

这样它才同时适合技术学习和面试讲述。

## 逐级推理链

### 1. Naïve：先把瓶颈看清楚

朴素 kernel 故意保持简单：每个线程负责一个输出元素，直接做内积。正因为简单，内存问题才会被看得很清楚：

- **A** 的一行会被反复读取
- **B** 的列访问在全局内存里局部性很差
- 计算公式好讲，但数据复用非常弱

它的价值在于告诉读者：在进入更复杂的 GPU 调度之前，SGEMM 首先是数据移动问题。

### 2. Tiled：用协作换复用

Tiling 把故事从“每个线程自己取数据”改成“整个 block 把可复用工作集 stage 到共享内存”。

这个改动背后的架构含义远大于代码表面：

- 全局内存访问更容易做到 coalesced
- 同一份 tile 会被很多次乘加重复利用
- 同步从性能细节变成正确性前提

从这一级开始，系统真正呈现出 GPU 架构味道，而不只是矩阵乘法公式。

### 3. Bank-Free：修共享内存的“形状”

一旦共享内存成为核心，布局就不能忽略。Bank-free kernel 存在的原因是：即使已经 tiled，共享内存访问仍然可能因为 bank pattern 而浪费周期。

通过给共享 tile 做 padding，可以改变地址步长，让常见访问模式不再挤进同一个 bank。即便示例 benchmark 没有总是表现出巨大提速，这个推理依然重要：布局更稳健，也更容易作为系统设计决策来解释。

### 4. Double Buffer：把 tile 循环变成调度问题

当共享内存复用和 bank 布局都处理好之后，下一个问题就变成了时间安排：能不能在当前 tile 还在被消费时，就开始准备下一块数据？

双缓冲的回答是把一个 tile buffer 变成两个交替角色：

- **当前缓冲区** 负责喂给计算
- **下一缓冲区** 负责提前加载下一轮数据

到这里，阶梯已经不只是内存布局问题，而是调度问题。

### 5. Tensor Core：加入带保护的快路径

Tensor Core 被放在最后一级，是因为它依赖前面的推理基础：

- 读者已经理解为什么数据移动重要
- 项目可以把 WMMA 和强 FP32 基线放在一起对比
- 实现可以解释为什么“更快的路径”也必须受保护

仓库把 WMMA 视为受约束的加速路径，而不是普适替代品。友好 shape 可以走 compute-only WMMA benchmark；不支持的 shape 必须老老实实回退。

## 这条阶梯真正想教什么

| 结论 | 出现位置 |
|------|----------|
| 全局内存访问模式决定最初级行为 | Naïve → Tiled |
| 共享内存有价值，但布局也必须正确 | Tiled → Bank-Free |
| 局部性改善后，调度与重叠才开始重要 | Bank-Free → Double Buffer |
| 更高吞吐硬件会引入 API 与 shape 约束 | Double Buffer → Tensor Core |
| 正确性、测量范围与 fallback 规则必须进入设计叙事 | 整个阶梯 |

## 如何谈性能而不过度简化

阶梯是一条推理链，不是“在所有机器上都严格单调增长”的数字承诺。例如：

- tiled 和 bank-free 可能因为 occupancy 与访问细节互有高低
- 双缓冲在已经能很好隐藏延迟的 GPU 上，收益可能只有温和提升
- Tensor Core 数字必须拆成端到端与 compute-only 两种视角

正确问题不是“哪一页数字最大”，而是“这一步在解决什么瓶颈，前提条件是什么？”

## 相关页面

- [架构概述](/zh/architecture/)
- [Memory Flow](/zh/architecture/memory-flow)
- [Tensor Core 路径](/zh/architecture/tensor-core-path)
- [Benchmark 范围](/zh/validation/benchmark-scope)
