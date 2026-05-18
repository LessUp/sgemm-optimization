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
| `launch_tensor_core_sgemm_with_fallback` | FP32 | 安全端到端 wrapper：检查设备与 16 对齐维度，转换为 FP16，不满足 WMMA 条件时回退到 FP32。 |
| `launch_tensor_core_sgemm_fp16` | FP16 | 纯计算路径：要求 `sm_70+` 和 16 对齐维度，不满足条件时抛错而不是回退。 |

这一区分让 benchmark 可以同时报告“包含转换/回退的端到端结果”和“纯 WMMA compute-only 结果”，避免把 fallback 当作 Tensor Core 计算性能。



## Benchmark 范围说明

仓库里常被引用的 **40.2% cuBLAS / 2300 GFLOPS**，对应的是 **WMMA compute-only** 测量：它只代表 Tensor Core 友好 shape 下纯 WMMA 路径的上界，不等同于包含转换与 fallback 的端到端 wrapper。

阅读本页时，请把下面两种数字严格分开：

| 范围 | 含义 | 如何解读 |
|------|------|----------|
| WMMA 端到端 | 安全 FP32 wrapper，包含 FP32→FP16 转换与 fallback 处理 | 用来和 FP32 kernel 做真实调用路径对比 |
| WMMA compute-only | 预转换 FP16 的纯 WMMA 快路径 | 用来观察原始 Tensor Core 计算上界 |

关于结果解释，请配合阅读 [Benchmark 结果](/zh/validation/benchmark-results) 与 [Benchmark 范围](/zh/validation/benchmark-scope)。



## 关键要点

1. **Tensor Core**：专用矩阵单元，~8× 理论峰值（~3-4× 实现）
2. **WMMA API**：Tensor Core 编程的 warp 级抽象
3. **混合精度**：FP16 输入，FP32 累加
4. **对齐**：M、K、N 必须是 16 的倍数才能使用 WMMA
5. **回退**：始终为边缘情况提供 FP32 kernel
