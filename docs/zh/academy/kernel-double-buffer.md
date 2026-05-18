---
title: 4. Double Buffer
---

# Kernel 4: Double Buffer

重叠内存加载与计算



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



## 关键要点

1. **双缓冲**：两个缓冲区在加载和计算角色间交替
2. **重叠**：通过边计算边加载隐藏内存延迟
3. **Ping-Pong**：用 `t % 2` 交替缓冲区索引
4. **权衡**：2× 共享内存换取更好的延迟隐藏
5. **同步**：单个屏障同时处理两个操作
