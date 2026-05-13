---
title: Benchmark 结果
---

# Benchmark 结果

代表性性能说明，非通用承诺



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



## Tensor Core 说明

Benchmark 报告：

- **WMMA 端到端**：安全 FP32 wrapper，包含转换和回退处理
- **WMMA 仅计算**：预转换 FP16 的纯计算路径，仅在 `M`、`K`、`N` 均为 16 的倍数时显示

当维度不适合 Tensor Core 时，实现回退到更安全的 FP32 路径，而非强制 WMMA。



## 相关参考

- [快速上手](/zh/getting-started)
- [学习路径](/zh/learning-path)
- [README 中的 Kernel 演进](https://github.com/LessUp/sgemm-optimization/blob/master/README.md)
