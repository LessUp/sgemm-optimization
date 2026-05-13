---
title: 2. Tiled Kernel
---

# Kernel 2: Tiled 实现

共享内存分块以实现更好的数据复用



## Naïve 的问题

在 naïve kernel 中：
- 计算一行 C 需要读取那一行 A **N 次**
- 计算一列 C 需要读取那一列 B **M 次**



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



## 性能特征

| 指标 | Naïve | Tiled | 改进 |
|------|-------|-------|------|
| **GFLOPS (1024³)** | 604 | 753 | **+25%** |
| **全局内存流量** | 2MNK | 2MNK/TILE | **-97%** |
| **共享内存** | 0 KB | ~8 KB | 新增 |
| **内存受限？** | 是 | 仍是 | — |



## 关键要点

1. **共享内存**比全局内存快 ~100×
2. **分块**通过数据复用减少全局内存带宽
3. 当连续线程读取连续地址时实现**合并访问**
4. 线程共享数据时需要**同步**
5. **模板参数**允许编译时选择 tile 大小
