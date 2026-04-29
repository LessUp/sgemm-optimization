---
layout: default
title: 5. Tensor Core
parent: 首页
nav_order: 7
permalink: /zh/docs/kernel-tensor-core/
lang: zh-CN
page_key: zh-kernel-tensor-core
lang_ref: kernel-tensor-core
---

# Kernel 5: Tensor Core (WMMA)
{: .fs-8 }

利用专用矩阵乘累加硬件
{: .fs-6 .fw-300 }

---

## 概述

NVIDIA Tensor Core 是执行**混合精度矩阵乘累加**操作的专用单元。单条指令可计算 4×4×4 或更大的矩阵操作 — 达到 CUDA 核心 **~8× 理论峰值吞吐量**（实践中 ~3-4×）。

<div class="highlight-box info">
  <strong>关键洞察</strong><br>
  Tensor Core 使用 FP16 输入和 FP32 累加。这种混合精度提供显著加速，同时对大多数深度学习和 HPC 工作负载保持精度。
</div>

---

## Tensor Core 架构

### 硬件能力

| 代数 | 架构 | 操作/周期 | 精度 |
|------|------|----------|------|
| Volta (V100) | sm_70 | 64 FMA | FP16/FP32 |
| Turing (RTX 20) | sm_75 | 64 FMA | FP16/INT8/INT32 |
| Ampere (A100/RTX 30) | sm_80/sm_86 | 256 FMA | FP16/BF16/TF32 |
| Ada (RTX 40) | sm_89 | 512 FMA | FP16/BF16/TF32 |
| Hopper (H100) | sm_90 | 1024 FMA | FP8/FP16/BF16 |

### WMMA 片段大小

```
Warp 矩阵乘累加 (WMMA)：

片段 A: 16×16 FP16 矩阵（行主序）
片段 B: 16×16 FP16 矩阵（行主序）
片段 C: 16×16 FP32 矩阵（行主序）
             ↓
          D = A × B + C
             ↓
片段 D: 16×16 FP32 矩阵

一个 warp（32 线程）协作完成一个 16×16×16 操作。
```

---

## WMMA API

### 片段声明

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// 矩阵片段
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;
fragment<accumulator, 16, 16, 16, float> d_frag;
```

### WMMA 操作

```cpp
// 初始化累加器为零
fill_fragment(c_frag, 0.0f);

// 从全局/共享内存加载数据
load_matrix_sync(a_frag, A_ptr, lda);
load_matrix_sync(b_frag, B_ptr, ldb);

// 执行矩阵乘累加
mma_sync(d_frag, a_frag, b_frag, c_frag);

// 存储结果
store_matrix_sync(D_ptr, d_frag, ldd, mem_row_major);
```

---

## 安全包装器与纯 WMMA 路径

仓库中 Tensor Core 有两个不同的接口语义：

| 接口 | 输入 | 行为 |
|------|------|------|
| `launch_tensor_core_sgemm` | FP32 | 安全端到端 wrapper：检查设备与 16 对齐维度，转换为 FP16，不满足 WMMA 条件时回退到 FP32。 |
| `launch_tensor_core_sgemm_fp16` | FP16 | 纯计算路径：要求 `sm_70+` 和 16 对齐维度，不满足条件时抛错而不是回退。 |

这一区分让 benchmark 可以同时报告“包含转换/回退的端到端结果”和“纯 WMMA compute-only 结果”，避免把 fallback 当作 Tensor Core 计算性能。

---

## 混合精度考量

### 精度权衡

```
FP32 精度：~7 位十进制数字
FP16 精度：~3 位十进制数字

FP32 → FP16 转换引入量化误差。
但 FP32 累加保持求和精度。
```

### 验证容差

```cpp
// 标准 FP32 kernel
const float rtol_fp32 = 1e-3f;
const float atol_fp32 = 1e-4f;

// Tensor Core 混合精度
const float rtol_tc = 5e-2f;   // 宽松 50×
const float atol_tc = 1e-2f;   // 宽松 100×
```

### FP16 何时可接受

| 用例 | FP16 可行？ |
|------|------------|
| 深度学习训练 | ✓ 是 |
| 深度学习推理 | ✓ 是 |
| 科学计算 | ⚠️ 需检查 |
| 金融计算 | ✗ 否 |

---

## 性能特征

| 指标 | Double Buffer | Tensor Core | 改进 |
|------|---------------|-------------|------|
| **GFLOPS (1024³)** | 701 | 2300 | **3.3×** |
| **vs cuBLAS** | 12.2% | 40.2% | — |
| **精度** | FP32 | FP16→FP32 | 混合 |
| **对齐要求** | 无 | 纯 WMMA 需要 16×；wrapper 不满足时回退 | 有 |
| **计算单元** | CUDA Core | Tensor Core | 专用 |

---

## 优化机会

我们的 Tensor Core kernel 只达到 cuBLAS 性能的 **40%**。差距来自：

1. **多级分块**：Warp 级 + 线程级 tile
2. **指令流水线**：并发发出多个 MMA
3. **共享内存暂存**：更好的 FP16 数据布局
4. **收尾融合**：将输出处理与 MMA 合并

生产使用时，**始终使用 cuBLAS** 或 **CUTLASS**。

---

## 关键要点

1. **Tensor Core**：专用矩阵单元，~8× 理论峰值（~3-4× 实现）
2. **WMMA API**：Tensor Core 编程的 warp 级抽象
3. **混合精度**：FP16 输入，FP32 累加
4. **对齐**：M、K、N 必须是 16 的倍数才能使用 WMMA
5. **回退**：始终为边缘情况提供 FP32 kernel
