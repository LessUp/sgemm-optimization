---
layout: default
title: Benchmark 结果
nav_order: 9
permalink: /zh/docs/benchmark-results/
lang: zh-CN
page_key: zh-benchmark-results
lang_ref: benchmark-results
---

# Benchmark 结果
{: .fs-8 }

代表性性能说明，非通用承诺
{: .fs-6 .fw-300 }

---

## 先读这个

性能取决于 GPU 型号、CUDA 版本、时钟、温度和矩阵形状。将下方数字视为帮助理解优化阶梯的**参考快照**，而非每台机器的保证。

---

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

---

## 比精确数字更重要的事

| 过渡 | 主要教训 |
|------|----------|
| Naive → Tiled | 共享内存复用立即可见效果 |
| Tiled → Bank-Free | 内存布局细节可消除隐藏瓶颈 |
| Bank-Free → Double Buffer | 重叠和分阶段在内存停滞主导时有帮助 |
| Double Buffer → Tensor Core | 专用硬件显著改变上限 |

---

## Tensor Core 说明

Benchmark 报告：

- **WMMA 端到端**：安全 FP32 wrapper，包含转换和回退处理
- **WMMA 仅计算**：预转换 FP16 的纯计算路径，仅在 `M`、`K`、`N` 均为 16 的倍数时显示

当维度不适合 Tensor Core 时，实现回退到更安全的 FP32 路径，而非强制 WMMA。

---

## 在你的机器上复现

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

./build/bin/sgemm_benchmark -a
./build/bin/sgemm_benchmark --dims 256 384 640
```

如需更长测量时间：

```bash
./build/bin/sgemm_benchmark -a --warmup 10 --benchmark 50
```

---

## 相关参考

- [快速上手](getting-started/)
- [学习路径](learning-path/)
- [README 中的 Kernel 演进](https://github.com/LessUp/sgemm-optimization/blob/master/README.md)
