---
layout: default
title: 1. Naïve Kernel
parent: 首页
nav_order: 2
permalink: /zh/docs/kernel-naive
lang: zh-CN
page_key: zh-kernel-naive
lang_ref: kernel-naive
---

# Kernel 1: Naïve 实现
{: .fs-8 }

最简单的方法 — 每个线程计算一个输出元素
{: .fs-6 .fw-300 }

---

## 概述

Naïve kernel 是我们的**基线实现**。它采用最直接的矩阵乘法方法：每个 CUDA 线程负责计算输出矩阵 C 的恰好一个元素。

<div class="highlight-box info">
  <strong>学习目标</strong><br>
  理解基本 CUDA 编程模型，识别为什么这种"显而易见"的方法在 GPU 上性能很差。
</div>

---

## 算法

对于矩阵乘法 C = A × B，其中：
- A 是 M × K
- B 是 K × N  
- C 是 M × N

每个线程计算：
```
C[row, col] = Σ A[row, k] × B[k, col]  for k = 0 to K-1
```

### 线程映射

```
Grid:  (N + 15) / 16 × (M + 15) / 16 个 block
Block: 16 × 16 个线程

Block (bx, by) 中的 Thread (tx, ty) 计算：
  row = by × 16 + ty
  col = bx × 16 + tx
  C[row, col] 如果 row < M 且 col < N
```

---

## 实现

```cpp
// 文件: src/kernels/naive_sgemm.cuh

__global__ void sgemm_naive_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    // 计算全局行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // 计算行和列的点积
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        // 写入结果
        C[row * N + col] = sum;
    }
}
```

---

## 内存访问模式

### 问题：非合并访问

```
Thread (0,0) 读取: A[0,0], A[0,1], A[0,2] ...  →  连续 ✓
                   B[0,0], B[N,0], B[2N,0] ...   →  步长-N ✗

Thread (0,1) 读取: A[0,0], A[0,1], A[0,2] ...  →  与线程 0 相同！
                   B[0,1], B[N,1], B[2N,1] ...   →  步长-N ✗
```

读取矩阵 **B** 时，连续线程访问间隔 **N** 个 float 的元素（步长-N 访问）。这导致：

1. **内存请求串行化** — GPU 必须发出单独的加载
2. **缓存效率低** — 加载的数据不能在线程间共享
3. **~12.5% 带宽利用率** — vs 合并访问的 100%

---

## 性能特征

| 指标 | 值 | 分析 |
|------|-----|------|
| **GFLOPS (1024³)** | ~604 | 基线 |
| **内存访问** | B 矩阵步长访问 | 可能 bank 冲突 |
| **数据复用** | 无 | 每个元素每次使用读取一次 |
| **计算强度** | 2 FLOPs / 8 bytes | 内存受限 |

### Roofline 位置

位于**内存受限区域**深处 — 性能完全受内存带宽限制，而非计算能力。

---

## 下一步

Naïve kernel 的主要瓶颈是**内存带宽**。下一个 kernel 将通过**共享内存分块**来解决这个问题：

→ 继续阅读 [Tiled Kernel](kernel-tiled){: .btn .btn-primary }

---

## 延伸阅读

- [CUDA 最佳实践指南 — 内存合并](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
- [NVIDIA 博客: CUDA Pro Tip](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)
