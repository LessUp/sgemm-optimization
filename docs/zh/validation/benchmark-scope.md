---
title: Benchmark 范围
---

# Benchmark 范围

本页定义 benchmark 标签的含义，以及读者应如何解释这些数字。

## 规范标签拆分

Benchmark 套件刻意区分两种 Tensor Core 视角：

- **WMMA 端到端**：面向安全 FP32 调用路径，包含 FP32→FP16 转换和 fallback 行为
- **WMMA compute-only**：纯预转换 WMMA 计算路径，仅在 `M`、`K`、`N` 为正且都是 16 的倍数时显示

之所以把这两个标签分开，是因为它们回答的是不同问题。

## 每个标签能证明什么

| 标签 | 能证明什么 | 单独看时不能证明什么 |
|------|------------|----------------------|
| cuBLAS | 当前 GPU 与工具链下的参考吞吐 | 跨环境评价项目 kernel 设计质量 |
| 标准 FP32 kernels | 仓库 FP32 路径在选定 shape 上的端到端行为 | Tensor Core 潜力 |
| WMMA 端到端 | 真实调用者通过安全 Tensor Core wrapper 看到的效果 | Tensor Core 峰值计算吞吐 |
| WMMA compute-only | 兼容维度下纯 WMMA 计算路径的上界 | 转换、fallback 或不规则 shape 的代价 |

## 读者应如何解释公开数字

1. **把它们当作代表性证据，而不是通用承诺。** 一份公开快照只记录了某个 GPU、某个 CUDA 栈和某组 benchmark 配置。
2. **必须同类对比。** 不要把“仅对齐 shape 的 compute-only 数字”与“混合 shape 的端到端数字”直接比较却不标注。
3. **默认存在硬件敏感性。** Volta、Turing、Ampere、Ada、Hopper 的主导瓶颈不会完全一样。
4. **默认这些数字不是 CI 跑出来的。** 托管 CI 证明仓库健康，不证明 benchmark 真实表现。

## 规范 benchmark 集合

CLI 默认值也是信任模型的一部分：

- 未显式给维度时，默认单案例是 `1024 x 1024 x 1024`
- `-a` 会展开为 `512x512x512`、`1024x1024x1024`、`256x384x640`、`511x513x1025`

之所以这样设计，是为了让仓库同时报告友好 shape 和棘手 shape，而不是把它们假装成同一种工作负载。

## 引用数字前请先确认

- 给出 GPU 与 CUDA 环境
- 给出完整命令
- 说明是端到端还是 compute-only
- 说明是一组单案例还是混合默认集
- 如需复跑，引导读者阅读 [可复现性](/zh/validation/reproducibility)
