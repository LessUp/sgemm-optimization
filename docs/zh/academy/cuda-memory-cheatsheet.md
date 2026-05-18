---
title: CUDA 内存速查表
---

# CUDA 内存速查表

这张速查表现在被纳入 [资源中心](/zh/research/)。当你要重新打开 kernel 代码、profiler 输出或 WMMA 约束时，它可以帮你先把关键内存问题对齐。

## 这页最适合什么时候看

- 重新阅读 [Memory Flow](/zh/architecture/memory-flow) 或 [分块内核](/zh/academy/kernel-tiled) 之前。
- benchmark 结果变了，想先做一轮快速内存检查，而不是直接怪 occupancy 或 Tensor Cores 时。
- 面试或评审场景里，需要快速解释内存行为，但不想立刻翻完整 CUDA 手册时。

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

## Shared memory 观察点

| 问题 | 为什么重要 |
|---|---|
| 线程写入的 tile 布局，后续读取时还是连续的吗？ | shared memory 只有在它确实修复了全局内存访问问题时才有价值。 |
| padding 或索引重排，是否真的消除了热点路径上的 bank conflict？ | 如果 bank conflict 还在，tile 设计的收益很可能被吃掉。 |
| shared memory 占用增加后，occupancy 是否被压得过低？ | 有些分块收益会被更紧的 launch 几何限制抵消。 |

## Tensor Core 内存提示

| 主题 | 记住这点 |
|---|---|
| 对齐约束 | WMMA 路径通常要求维度按片段友好的大小对齐，常见是 16。 |
| 数据转换 | 端到端耗时包含转换和 wrapper 逻辑，而不只是矩阵乘本身。 |
| 安全行为 | 不友好 shape 应回退到 FP32 路径，而不是强行走 WMMA 导致结论失真。 |
| 结果汇报 | 对比实现前，先区分端到端与仅计算数据。 |

## 读 kernel 的快速清单

1. 能否解释一个 warp 的全局内存访问顺序？
2. shared memory 布局是真的在降低冲突，而不是只把数据搬了个地方吗？
3. 寄存器累加器是否受控且必要？
4. Tensor Core 回退行为是否清晰？
5. benchmark 标签是否和真实测量路径一致？

## 下一步去哪里

- [资源中心](/zh/research/)
- [延伸阅读路线](/zh/research/further-reading)
- [参考资料清单](/zh/research/references)
- [诊断闭环](/zh/academy/diagnosis-loop)
