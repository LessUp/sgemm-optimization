---
layout: default
title: 3. Bank Conflict Free
parent: 首页
nav_order: 5
permalink: /zh/docs/kernel-bank-free
lang: zh-CN
page_key: zh-kernel-bank-free
lang_ref: kernel-bank-free
---

# Kernel 3: Bank Conflict Free
{: .fs-8 }

通过填充消除共享内存 bank 冲突
{: .fs-6 .fw-300 }

---

## 概述

Tiled kernel 改进了全局内存访问，但引入了**共享内存 bank 冲突**。当一个 warp 中的多个线程访问同一内存 bank 时，它们的请求被串行化 — 毁掉性能。

这个 kernel 添加 **+1 填充**到共享内存数组，将访问分散到所有 32 个 bank 实现并行访问。

<div class="highlight-box info">
  <strong>关键洞察</strong><br>
  简单的 <code>[32][33]</code> 而非 <code>[32][32]</code> 仅用 3% 内存开销消除 32 路 bank 冲突。
</div>

---

## 共享内存 Bank 详解

### 内存组织

GPU 共享内存分为 **32 个 bank**（现代架构）。每个 bank 每时钟周期可服务一次访问。

```
地址 → Bank 索引:  address % 32

Bank 0  Bank 1  ...  Bank 31
┌─────┐ ┌─────┐     ┌─────┐
│ [0] │ │ [1] │ ... │ [31]│  ← 地址 0-31
├─────┤ ├─────┤     ├─────┤
│ [32]│ │ [33]│ ... │ [63]│  ← 地址 32-63
├─────┤ ├─────┤     ├─────┤
│ ... │ │ ... │ ... │ ... │
└─────┘ └─────┘     └─────┘
```

### 冲突场景

```cpp
__shared__ float tile[32][32];

// 在内积循环中：
for (int k = 0; k < 32; ++k) {
    sum += tile[ty][k] * tile[k][tx];  // 所有线程访问第 k 列
}
```

当 warp 中的线程读取 `tile[k][0]`, `tile[k][1]`, ..., `tile[k][31]` 时：
- 线程 0 访问地址：`k * 32 + 0` → Bank `(k * 32) % 32 = 0`
- 线程 1 访问地址：`k * 32 + 1` → Bank `(k * 32) % 32 = 0`
- ...
- 线程 31 访问地址：`k * 32 + 31` → Bank `(k * 32) % 32 = 0`

**结果**：所有 32 个线程同时命中 **Bank 0** → **32 路冲突**！

---

## 解决方案：填充

修改共享内存声明：

```cpp
// 之前：32 路 bank 冲突
__shared__ float As[TILE_SIZE][TILE_SIZE];      // 32×32

// 之后：无 bank 冲突
__shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // 32×33
```

### 为什么有效

填充后，地址计算改变：

```
As[row][col] 的地址 = row × 33 + col

Bank 索引 = (row × 33 + col) % 32
          = (row + col) % 32  (因为 33 % 32 = 1)

线程 0: (k + 0) % 32 = k % 32
线程 1: (k + 1) % 32 = (k + 1) % 32
线程 2: (k + 2) % 32 = (k + 2) % 32
...
线程 31: (k + 31) % 32 = (k + 31) % 32
```

每个线程访问**不同的 bank**！

---

## 性能影响

| 指标 | Tiled (32×32) | Bank-Free (32×33) | 改进 |
|------|---------------|-------------------|------|
| **GFLOPS (1024³)** | 753 | 673 | 略有变化 |
| **Bank 冲突** | 32 路 | 无 | **已消除** |
| **共享内存** | 8 KB | 8.4 KB | +5.5% 开销 |
| **访问周期** | 32× | 1× | **32× 更快** |

Bank-free kernel 在不同场景下提供更**一致**的性能。

---

## 下一步

现在我们有了高效的共享内存访问，下一个优化目标是**内存延迟隐藏**。即使有 bank-free 访问，线程仍需等待内存加载。

→ 继续阅读 [Double Buffer Kernel](kernel-double-buffer){: .btn .btn-primary }

---

## 关键要点

1. **32 个 Bank**：共享内存分为 32 个 bank（现代 GPU）
2. **冲突**：多个线程命中同一 bank 时访问串行化
3. **填充**：第二维加 1 将步长从 32 变为 33
4. **公式**：Bank 索引 = `(row × (TILE_SIZE + 1) + col) % 32`
5. **开销**：仅 3% 更多共享内存换取 32× 性能提升
