---
layout: default
title: 4. Double Buffer
parent: 首页
nav_order: 6
permalink: /zh/docs/kernel-double-buffer/
lang: zh-CN
page_key: zh-kernel-double-buffer
lang_ref: kernel-double-buffer
---

# Kernel 4: Double Buffer
{: .fs-8 }

重叠内存加载与计算
{: .fs-6 .fw-300 }

---

## 概述

双缓冲（或 "ping-pong 缓冲"）技术将**全局内存加载**与**共享内存计算**重叠。在计算一个 tile 时，我们加载下一个 tile — 隐藏内存延迟。

<div class="highlight-box info">
  <strong>关键洞察</strong><br>
  现代 GPU 可以并发执行内存操作和计算。双缓冲利用这一点在等待内存时保持 ALU 繁忙。
</div>

---

## 问题：顺序执行

在 tiled kernel 中：

```
时间线：
  加载 Tile 0 ──────────────────▶
                                计算 Tile 0 ───────────────▶
                                                              加载 Tile 1 ───▶
                                                                              计算 Tile 1 ──▶

问题：加载期间 GPU 空闲，计算期间内存空闲
```

---

## 解决方案：双缓冲

使用**两个共享内存缓冲区**交替角色：
- **缓冲区 0**：正在计算
- **缓冲区 1**：正在从全局内存加载

```
双缓冲时间线：
  加载 Tile 0 ──────────────────▶
  加载 Tile 1 ───────────────────────▶
          计算 Tile 0 ─────────────▶
          加载 Tile 2 ───────────────────────▶
                      计算 Tile 1 ─────────▶
                      加载 Tile 3 ───────────────────▶
                                  计算 Tile 2 ─────▶

结果：计算与加载重叠！
```

---

## 共享内存布局

```cpp
// 单缓冲（之前）
__shared__ float As[TILE_SIZE][TILE_SIZE + 1];
__shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

// 双缓冲（之后）
__shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];  // [2] 用于 ping-pong
__shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];
```

**权衡**：2× 共享内存使用换取延迟隐藏。

---

## 实现要点

```cpp
// 文件: src/kernels/double_buffer_sgemm.cuh

template<int TILE_SIZE = 32>
__global__ void sgemm_double_buffer_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    // 双缓冲 ping-pong
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // 预加载第一个 tile
    load_tile(0, 0);
    __syncthreads();

    // 双缓冲主循环
    for (int t = 0; t < num_tiles; ++t) {
        int curr = t % 2;        // 当前缓冲区用于计算
        int next = (t + 1) % 2;  // 下一个缓冲区用于加载

        // 异步加载下一个 tile（如果存在）
        if (t + 1 < num_tiles) {
            load_tile(next, t + 1);
        }

        // 在当前 tile 上计算
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[curr][ty][k] * Bs[curr][k][tx];
        }

        __syncthreads();  // 等待计算和下一次加载完成
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

---

## 性能特征

| 指标 | Bank-Free | Double Buffer | 改进 |
|------|-----------|---------------|------|
| **GFLOPS (1024³)** | 673 | 701 | **+4%** |
| **共享内存** | 8.4 KB | 16.8 KB | 2× |
| **寄存器压力** | 低 | 中等 | — |
| **Occupancy** | 较高 | 较低 | 权衡 |

<div class="highlight-box warning">
  <strong>注意</strong><br>
  性能改进（~4%）不大，因为现代 GPU 通过 warp 调度有有效的内存延迟隐藏。双缓冲对计算最少的内存受限 kernel 更有影响。
</div>

---

## 下一步

我们已优化：
- ✓ 全局内存合并
- ✓ 共享内存 bank 冲突
- ✓ 内存延迟隐藏

最后的边界：**专用矩阵硬件**。现代 GPU 有 Tensor Core，可以在一个周期内执行 4×4×4 矩阵乘累加。

→ 继续阅读 [Tensor Core Kernel](kernel-tensor-core/){: .btn .btn-primary }

---

## 关键要点

1. **双缓冲**：两个缓冲区在加载和计算角色间交替
2. **重叠**：通过边计算边加载隐藏内存延迟
3. **Ping-Pong**：用 `t % 2` 交替缓冲区索引
4. **权衡**：2× 共享内存换取更好的延迟隐藏
5. **同步**：单个屏障同时处理两个操作
