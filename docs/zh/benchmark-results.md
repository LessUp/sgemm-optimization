---
title: Benchmark 结果
---

# Benchmark 结果

这是代表性的性能快照，不是通用承诺。

本页现在只保留**结果快照**职责。关于信任边界与解释规则，请阅读 [验证](/zh/validation/)。关于实验设计，请阅读 [方法论](/zh/methodology/)。

## 参考快照

RTX 3060 Laptop 在 `1024 x 1024 x 1024` 的示例数据：

| Kernel | GFLOPS | vs cuBLAS |
|--------|-------:|----------:|
| cuBLAS | 5727 | 100.0% |
| Tensor Core (WMMA compute-only) | 2300 | 40.2% |
| Tiled | 753 | 13.1% |
| Double Buffer | 701 | 12.2% |
| Bank-Free | 673 | 11.8% |
| Naive | 604 | 10.6% |

## 如何阅读本页

- 这些数字是**本地代表性快照**，不是所有 GPU 上的承诺。
- 在比较它们之前，先阅读 [Benchmark 范围](/zh/validation/benchmark-scope)。
- 把 `WMMA compute-only` 看成窄范围快路径标签，而不是端到端行为的替代品。
- 默认这些数字**不是**托管 CI 证明出来的；只有本地 GPU 运行才能证明它们。

## Tensor Core 说明

Benchmark 套件会报告：

- **WMMA 端到端**：安全 FP32 wrapper，包含转换和 fallback 处理
- **WMMA compute-only**：预转换 FP16 的纯 WMMA 计算路径，仅在 `M`、`K`、`N` 为 16 的倍数时显示

当维度不适合 Tensor Core 时，实现会回退到更安全的 FP32 路径，而不是强行启用 WMMA。

## 建议配套阅读

- [验证概览](/zh/validation/)
- [正确性策略](/zh/validation/correctness-policy)
- [Benchmark 范围](/zh/validation/benchmark-scope)
- [可复现性](/zh/validation/reproducibility)
- [方法论](/zh/methodology/)
