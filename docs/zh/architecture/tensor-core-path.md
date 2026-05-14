---
title: Tensor Core 路径
---

# Tensor Core 路径

Tensor Core 支持在这里被当成一条“带保护的快路径”。仓库不会假设所有矩阵 shape、所有 GPU、所有 benchmark 标签都应该走 WMMA。

## 两条公开路径

| 路径 | 输入类型 | 典型用途 | 不支持时的行为 |
|------|----------|----------|----------------|
| `launch_tensor_core_sgemm_with_fallback` | FP32 | 安全端到端执行与公开 benchmark | 回退到显式 FP32 kernel |
| `launch_tensor_core_sgemm_fp16` | FP16 | compute-only WMMA benchmark | 当设备或维度不满足条件时抛错 |

这种拆分是架构叙事的核心，因为它避免把转换成本和 fallback 行为误报成纯 Tensor Core 计算速度。

## 路径选择逻辑

### 1. 设备 guard

WMMA 需要 Tensor Core 能力（`sm_70+`）。如果设备不支持，安全 FP32 wrapper 会直接停留在 fallback 路径。

### 2. Shape guard

WMMA 路径要求维度与 fragment 形状对齐。本仓库中这意味着 `M`、`K`、`N` 都必须是 16 的倍数。

### 3. 数据格式 guard

安全公共入口从 FP32 输入开始，因此必须先分配 FP16 staging buffer，并把两份输入矩阵转换后再发射 FP16 WMMA kernel。

### 4. Fallback 策略

当 guard 检查失败时，仓库不会伪装成 Tensor Core 执行。它会调用由调用者显式传入的 fallback。在本仓库的 benchmark 与推荐 helper 路径中，这个 fallback 通常是 bank-conflict-free FP32 kernel。

## 为什么 fallback 很重要

Fallback 不是微不足道的实现细节，它让三个架构承诺成立：

- **不规则 shape 仍然有正确性路径**。
- **benchmark 标签保持诚实**，因为不支持的情况不会被算成 WMMA 胜利。
- **Tensor Core 模块保持清晰边界**，因为调用者必须思考“不支持时怎么办”。

## 实际受保护流程

```text
FP32 输入
  ↓
检查设备能力与 16 对齐维度
  ├─ 不支持 → 显式 FP32 fallback
  └─ 支持
       ↓
    把 A、B 从 FP32 转成 FP16
       ↓
    启动 WMMA fast path
       ↓
    以 FP32 累加得到输出 C
```

compute-only benchmark 会从这条流程的更后面开始：它假设输入已经是 FP16，并跳过转换与 fallback。

## Shape 约束与报告纪律

| 问题 | 架构回答 |
|------|----------|
| 所有矩阵都能用 WMMA 吗？ | 不能，必须是友好维度。 |
| 端到端和 compute-only 数字能混为一谈吗？ | 不能。前者包含转换/fallback，后者隔离 WMMA 计算本身。 |
| Tensor Core 会取代 FP32 kernel 吗？ | 不会。它是在 FP32 体系上附加的一条受约束快路径。 |
| 仓库通常接入什么 fallback？ | Bank-conflict-free FP32 kernel helper。 |

## 为什么仍然值得保留 WMMA

即便当前实现是教学型，而不是 cuBLAS 级极限优化，Tensor Core 路径仍然提供了最后一个重要架构教训：

- 专用硬件确实能抬高吞吐上限
- 更高吞吐会带来 shape、API 与精度约束
- 一个可信系统必须解释什么时候走快路径，什么时候主动放弃

## 建议配套阅读

- [架构概述](/zh/architecture/)
- [Kernel 阶梯](/zh/architecture/kernel-ladder)
- [Benchmark 范围](/zh/validation/benchmark-scope)
- [Tensor Core WMMA](/zh/kernel-tensor-core)
