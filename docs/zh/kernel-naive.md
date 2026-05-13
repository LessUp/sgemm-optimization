---
title: 1. Naïve Kernel
---

# Kernel 1: Naïve 实现

最简单的方法 — 每个线程计算一个输出元素



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



## 下一步

Naïve kernel 的主要瓶颈是**内存带宽**。下一个 kernel 将通过**共享内存分块**来解决这个问题：

→ 继续阅读 [Tiled Kernel](/zh/kernel-tiled)

---
