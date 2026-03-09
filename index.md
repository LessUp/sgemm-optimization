---
layout: default
title: SGEMM Optimization
---

# SGEMM Optimization: From Naive to Tensor Core

从零手写极致优化的矩阵乘法 — HPC 领域的 "Hello World"。渐进式优化 CUDA SGEMM，展示 GPU 编程核心优化技术。

## 实测性能 (RTX 3060 Laptop, 1024×1024)

| Kernel | GFLOPS | vs cuBLAS |
|--------|--------|-----------|
| cuBLAS (参考) | 5727 | 100% |
| Tensor Core (WMMA) | 2300 | 40.2% |
| Tiled (32×32) | 753 | 13.1% |
| Double Buffer | 701 | 12.2% |
| Bank Conflict Free | 673 | 11.8% |
| Naive | 604 | 10.6% |

## 优化路线

| 版本 | 关键技术 |
|------|----------|
| Naive | 每线程计算一个输出元素 |
| Tiled | 共享内存分块，数据复用 |
| Bank Conflict Free | 共享内存 padding (+1) |
| Double Buffer | 计算与访存重叠 |
| Tensor Core | WMMA API，FP16→FP32 |

## 快速开始

```bash
# 编译 (根据 GPU 架构调整)
make GPU_ARCH=sm_86

# 运行基准测试
./build/sgemm_benchmark

# 或直接
make benchmark
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | CUDA C++17 |
| 构建 | Makefile |
| 依赖 | cuBLAS, Google Test (可选) |
| GPU | SM 70+ (Volta → Hopper) |

## 链接

- [GitHub 仓库](https://github.com/LessUp/sgemm-optimization)
- [README](README.md)
