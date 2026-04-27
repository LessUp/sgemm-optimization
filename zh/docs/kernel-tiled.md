---
layout: default
title: 2. Tiled Kernel
parent: 首页
nav_order: 4
permalink: /zh/docs/kernel-tiled/
lang: zh-CN
page_key: zh-kernel-tiled
lang_ref: kernel-tiled
---

# Kernel 2: Tiled 实现
{: .fs-8 }

共享内存分块以实现更好的数据复用
{: .fs-6 .fw-300 }

---

## 概述

Tiled kernel 引入**共享内存**大幅减少全局内存流量。不再让每个线程从全局内存加载 K 次数据，我们一次性加载 tile 到快速共享内存并复用它们。

<div class="highlight-box info">
  <strong>关键洞察</strong><br>
  A 和 B 的每个元素分别被使用 N 次和 M 次。共享内存将全局内存读取减少 <strong>TILE_SIZE×</strong>。
</div>

---

## Naïve 的问题

在 naïve kernel 中：
- 计算一行 C 需要读取那一行 A **N 次**
- 计算一列 C 需要读取那一列 B **M 次**

---

## 解决方案：分块 (Tiling)

将矩阵划分为 **TILE_SIZE × TILE_SIZE** 的块：

```
A (M×K)          B (K×N)          C (M×N)
┌───┬───┐       ┌───┬───┐       ┌───┬───┐
│A00│A01│       │B00│B01│       │C00│C01│
├───┼───┤   ×   ├───┼───┤   =   ├───┼───┤
│A10│A11│       │B10│B11│       │C10│C11│
└───┴───┘       └───┴───┘       └───┴───┘

C00 = A00×B00 + A01×B10
```

C 的每个块通过从 A 和 B 加载对应块到共享内存来计算。

---

## 实现

```cpp
// 文件: src/kernels/tiled_sgemm.cuh

template<int TILE_SIZE = 32>
__global__ void sgemm_tiled_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    // 共享内存 tile
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // 线程索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 全局位置
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // 遍历 tile
    for (int t = 0; t < num_tiles; ++t) {
        // 计算 tile 位置
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;

        // 从 A 加载 tile（合并访问）
        if (row < M && a_col < K)
            As[ty][tx] = A[row * K + a_col];
        else
            As[ty][tx] = 0.0f;

        // 从 B 加载 tile（合并访问）
        if (b_row < K && col < N)
            Bs[ty][tx] = B[b_row * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();  // 等待所有加载完成

        // 计算 tile 乘法
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();  // 等待所有线程完成计算
    }

    // 写入结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

---

## 内存访问模式

### 之前 (Naïve)
```
全局内存读取 = M × N × K 元素 × 2 个矩阵
             = 2 × M × N × K 次读取
```

### 之后 (Tiled)
```
全局内存读取 = A tile 的 (M × K) + B tile 的 (K × N)
             = 每个 tile 迭代 K × (M + N) 次读取
             = O((M × N × K) / TILE_SIZE) 总计
减少倍数: TILE_SIZE×
```

---

## 性能特征

| 指标 | Naïve | Tiled | 改进 |
|------|-------|-------|------|
| **GFLOPS (1024³)** | 604 | 753 | **+25%** |
| **全局内存流量** | 2MNK | 2MNK/TILE | **-97%** |
| **共享内存** | 0 KB | ~8 KB | 新增 |
| **内存受限？** | 是 | 仍是 | — |

---

## 下一步

虽然我们改进了全局内存访问，但引入了新问题：**共享内存 bank 冲突**。当多个线程访问同一内存 bank 时，它们的请求被串行化。

→ 继续阅读 [Bank Conflict Free Kernel](kernel-bank-free/){: .btn .btn-primary }

---

## 关键要点

1. **共享内存**比全局内存快 ~100×
2. **分块**通过数据复用减少全局内存带宽
3. 当连续线程读取连续地址时实现**合并访问**
4. 线程共享数据时需要**同步**
5. **模板参数**允许编译时选择 tile 大小
