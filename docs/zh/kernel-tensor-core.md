---
title: 5. Tensor Core
---

# Kernel 5: Tensor Core (WMMA)

利用专用矩阵乘累加硬件



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



## 安全包装器与纯 WMMA 路径

仓库中 Tensor Core 有两个不同的接口语义：

| 接口 | 输入 | 行为 |
|------|------|------|
| `launch_tensor_core_sgemm` | FP32 | 安全端到端 wrapper：检查设备与 16 对齐维度，转换为 FP16，不满足 WMMA 条件时回退到 FP32。 |
| `launch_tensor_core_sgemm_fp16` | FP16 | 纯计算路径：要求 `sm_70+` 和 16 对齐维度，不满足条件时抛错而不是回退。 |

这一区分让 benchmark 可以同时报告“包含转换/回退的端到端结果”和“纯 WMMA compute-only 结果”，避免把 fallback 当作 Tensor Core 计算性能。



## 性能特征

| 指标 | Double Buffer | Tensor Core | 改进 |
|------|---------------|-------------|------|
| **GFLOPS (1024³)** | 701 | 2300 | **3.3×** |
| **vs cuBLAS** | 12.2% | 40.2% | — |
| **精度** | FP32 | FP16→FP32 | 混合 |
| **对齐要求** | 无 | 纯 WMMA 需要 16×；wrapper 不满足时回退 | 有 |
| **计算单元** | CUDA Core | Tensor Core | 专用 |



## 关键要点

1. **Tensor Core**：专用矩阵单元，~8× 理论峰值（~3-4× 实现）
2. **WMMA API**：Tensor Core 编程的 warp 级抽象
3. **混合精度**：FP16 输入，FP32 累加
4. **对齐**：M、K、N 必须是 16 的倍数才能使用 WMMA
5. **回退**：始终为边缘情况提供 FP32 kernel
