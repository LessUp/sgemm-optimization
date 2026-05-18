---
title: 性能模型
---

# 性能模型

本页描述 kernel 阶梯背后的分析性能模型，预测每个优化阶段的主导开销，并解释为什么每次架构改变会转移瓶颈。

## Roofline 框架

给定 GPU 上的 SGEMM 性能受两类资源限制：

1. **内存带宽** — 全局内存与 SM 之间的数据移动速率
2. **计算吞吐** — multiply-accumulate 操作的完成速率

**算术强度**（每字节 DRAM 流量对应的 FLOP 数）决定哪个上限是当前瓶颈：

```
I = FLOPs / 传输字节数
```

对于 N×N 矩阵乘法：
- FLOPs：2N³
- 朴素版 DRAM 流量：每个输出元素各读一次 A 和 B = 2N³ 数据元素 = O(N³) 字节
- Tiled 版 DRAM 流量：每个元素被读取 O(N/tile) 次 = O(N²) 字节 → 算术强度提升至 O(tile)

当 `I < 岭点` 时，kernel 受内存带宽限制；当 `I > 岭点` 时，受计算吞吐限制。

## 每阶段代价模型

### 朴素 FP32

```
算术强度 ≈ 1 FLOP/字节（FP32 密度）
状态：强内存带宽受限
瓶颈：每次 multiply-accumulate 都需要一次新的全局内存读取
```

朴素 kernel 不复用任何已加载的值。每个线程独立读取邻居线程也要读取的同一份 A、B 元素，导致大量冗余 DRAM 流量。

### Tiled FP32

```
算术强度 ≈ tile_size / 2 FLOPs/字节
状态：部分内存带宽受限（取决于 tile 大小）
瓶颈：共享内存带宽 + 同步开销
```

协作式加载到共享 tile 消除了大部分冗余 DRAM 流量。主导开销从全局内存带宽转移到共享内存吞吐和 `__syncthreads` 串行化。

### Bank-Free FP32

```
算术强度：与 tiled 相同（无新的复用）
状态：roofline 位置相同，有效共享内存延迟更低
瓶颈：残余共享内存 bank 冲突 → 通过 padding 消除
```

Padding 消除了 tiled 布局中的多路 bank 冲突。这不改变算术强度，但消除了一个降低有效带宽的共享内存串行化来源。

### Double Buffer

```
算术强度：与 tiled 相同
状态：在足够强力的硬件上变为计算受限
瓶颈：通过预取重叠隐藏内存延迟
```

双缓冲将 tile `k+1` 的取数与 tile `k` 的计算重叠。主导开销从内存延迟转移到计算吞吐，在支持该优化的硬件上接近计算上限。

### Tensor Core WMMA

```
算术强度：高（硬件 fragment 累加）
状态：计算受限，更高的吞吐上限
瓶颈：FP32→FP16 转换开销 + shape 约束
```

WMMA 指令每条完成一个 16×16×16 混合精度 multiply-accumulate，与 FP32 CUDA core 相比在 Tensor Core 硬件上提供 8× 更高的吞吐。实际开销是 FP32→FP16 转换和矩阵维度必须被 WMMA fragment 大小整除的要求。

## 性能预测与实测行为的对比

| Kernel | 预测主导开销 | 实测行为 |
|---|---|---|
| 朴素 | DRAM 带宽 | ✓ GFLOPS 极低，GPU 内存受限 |
| Tiled | 共享内存带宽 | ✓ 显著提升，对 tile 大小敏感 |
| Bank-Free | 减少共享内存串行化 | ✓ 在易冲突 shape 上比 tiled 有适度提升 |
| Double Buffer | 内存延迟（已隐藏） | ✓ 在高占用率 shape 上有提升 |
| Tensor Core | FP32→FP16 转换 + 计算 | ✓ 在能力保护下的大 shape 上有大幅提升 |

> 这些对比需要在本地 GPU 上执行。具体数字和硬件背景请参阅[Benchmark 结果](./benchmark-results)和[可复现性](./reproducibility)。

## 模型局限性

- Roofline 模型假设缓存行为完美；实际的占用率、warp 调度和 SM 资源限制会产生偏差。
- Tensor Core 路径中的 FP32→FP16 转换开销仅靠算术强度无法完整捕捉。
- 模型不考虑 L2 cache 效应，后者对中等规模矩阵结果影响显著。
- Tile 大小选择与占用率和寄存器文件压力之间的相互作用，超出了基本 roofline 模型的预测范围。

## 相关页面

- [Benchmark 范围](./benchmark-scope) — 覆盖哪些 shape 类别和硬件背景
- [Benchmark 结果](./benchmark-results) — 带硬件和 shape 标签的实测数字
- [可复现性](./reproducibility) — 如何负责任地运行和解释结果
- [Kernel 阶梯](../architecture/kernel-ladder) — 本代价模型的架构对应部分
- [Memory Flow](../architecture/memory-flow) — 算术强度转移背后的数据移动故事
