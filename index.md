---
layout: default
title: SGEMM Optimization
---

# SGEMM Optimization: From Naive to Tensor Core

[![CI](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/LessUp/sgemm-optimization/blob/main/LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)

从零手写极致优化的 CUDA 矩阵乘法 — HPC 领域的 "Hello World"。五个渐进式优化的 kernel 变体，展示 GPU 核心优化技术，从最朴素的三层循环到 **Tensor Core WMMA 达到 cuBLAS 40% 吞吐量**。

---

## 性能总览

在 NVIDIA GeForce RTX 3060 Laptop GPU (Ampere, sm_86) 上实测，矩阵维度 1024×1024×1024：

| Kernel | GFLOPS | vs cuBLAS | 耗时 | 核心技术 |
|--------|-------:|----------:|------:|----------|
| **cuBLAS** (参考) | 5727 | 100% | 0.375 ms | NVIDIA 高度优化库 |
| **Tensor Core** (WMMA) | 2300 | 40.2% | 0.934 ms | FP16→FP32 混合精度，硬件矩阵单元 |
| **Tiled** (32×32) | 753 | 13.1% | 2.853 ms | 共享内存分块，数据复用 |
| **Double Buffer** | 701 | 12.2% | 3.064 ms | 双缓冲流水线，计算/访存重叠 |
| **Bank Conflict Free** | 673 | 11.8% | 3.190 ms | 共享内存 padding 消除 bank 冲突 |
| **Naive** | 604 | 10.6% | 3.553 ms | 每线程一个输出元素，基线 |

*所有 kernel 均通过与 cuBLAS 的正确性验证 (allclose: rtol=1e-3, atol=1e-4; Tensor Core: rtol=5e-2)*

## 优化演进路线

```
  ┌─────────┐     ┌──────────┐     ┌──────────────┐     ┌───────────────┐
  │  Naive  │────▶│  Tiled   │────▶│  Bank-Free   │────▶│ Double Buffer │
  │ 604 GF  │     │ 753 GF   │     │   673 GF     │     │   701 GF      │
  └─────────┘     └──────────┘     └──────────────┘     └───────┬───────┘
                                                                │
                                                                ▼
                                                    ┌───────────────────┐
                                                    │   Tensor Core     │
                                                    │   2300 GF (WMMA)  │
                                                    └───────────────────┘
```

| 阶段 | 变更内容 | 为什么有效 |
|------|---------|-----------|
| **Naive → Tiled** | 将矩阵分块加载到共享内存 | 数据复用，全局内存流量降低 TILE_SIZE 倍 |
| **Tiled → Bank-Free** | 共享内存 padding `[32][33]` | 消除 32 路 bank conflict，共享内存带宽恢复 |
| **Bank-Free → Double Buffer** | 两个共享内存缓冲区交替使用 | 下一块加载与当前块计算重叠，掩盖内存延迟 |
| **→ Tensor Core** | WMMA API `mma_sync` | 专用矩阵计算单元，峰值性能 ~8× CUDA Core |

## 核心优化技术

### 内存合并访问 (Memory Coalescing)

Naive 版本访问矩阵 B 的列为非合并访问（stride = N），同一 warp 内线程访问不连续地址。Tiled 加载阶段确保 warp 级合并读取（stride = 1），带宽利用率从 ~12.5% 提升至接近 100%。

### 共享内存分块 (Tiling)

每个 tile 从全局内存加载一次，在共享内存中被复用 TILE_SIZE 次。全局内存访问复杂度从 O(N³) 降至 O(N³/TILE_SIZE)。共享内存延迟比全局内存低约 100×。

### Bank Conflict 消除

共享内存分为 32 个 bank。列访问时所有线程可能访问同一 bank，导致 32 路冲突串行化。通过 `+1` padding（`[32][33]`）使列访问跨越不同 bank，带宽恢复 ~32×。

### 双缓冲流水线 (Double Buffering)

使用两个共享内存缓冲区交替进行加载和计算。当前 tile 计算的同时预取下一个 tile，掩盖全局内存访问延迟。

### Tensor Core (WMMA API)

- 一个 warp (32 线程) 协作执行 16×16×16 矩阵乘法
- FP16 输入 + FP32 累加（混合精度）
- Ampere 架构上理论峰值约为 CUDA Core 的 8 倍
- 需要放宽验证容差（FP16 精度损失）

### Roofline 模型分析

SGEMM 算术强度 AI ≈ N/3（方阵 M=N=K）：

| 矩阵规模 | 算术强度 | 瓶颈类型 | 优化方向 |
|----------|---------|---------|---------|
| N < 256 | 低 | **内存受限** | 优化访存模式 |
| N = 512 | 中 | 过渡区域 | 兼顾两者 |
| N > 1024 | 高 | **计算受限** | 优化计算效率 |

## 快速开始

```bash
# Makefile 构建（调整 GPU 架构）
make GPU_ARCH=sm_86

# 或 CMake 构建
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 运行基准测试
make benchmark
# 或
./build/bin/sgemm_benchmark
```

## 项目结构

```
sgemm-optimization/
├── src/
│   ├── kernels/
│   │   ├── naive_sgemm.cuh              # Naive: 基础三层循环
│   │   ├── tiled_sgemm.cuh              # Tiled: 共享内存分块
│   │   ├── bank_conflict_free_sgemm.cuh # 消除 bank 冲突
│   │   ├── double_buffer_sgemm.cuh      # 双缓冲流水线
│   │   └── tensor_core_sgemm.cuh        # Tensor Core (WMMA)
│   ├── utils/
│   │   ├── cuda_utils.cuh               # CUDA 错误检查与工具函数
│   │   ├── benchmark.cuh                # 性能测试框架 (CUDA Events)
│   │   └── verify.cuh                   # 正确性验证 (vs cuBLAS)
│   └── main.cu                          # 主程序入口
├── tests/
│   └── test_sgemm.cu                    # Google Test 属性测试
├── roofline_data_*.csv                  # Roofline 分析原始数据
├── CMakeLists.txt                       # CMake 构建 (推荐)
└── Makefile                             # Make 构建 (快速上手)
```

## 测试与验证

基于 Google Test 的属性测试覆盖：

| 属性 | 验证内容 |
|------|---------|
| **数值正确性** | 所有 kernel 与 cuBLAS 结果一致 (allclose) |
| **Tensor Core 容差** | 在放宽 FP16 容差下结果正确 |
| **错误检测** | 验证系统能正确捕获注入的计算错误 |
| **维度不变性** | 所有 kernel 支持任意对齐矩阵维度 |

```bash
# 构建并运行测试
make test
# 或
cmake --build build --target test_sgemm && ctest --test-dir build
```

## 技术栈

| 类别 | 技术 |
|-----|------|
| **语言** | CUDA C++17 |
| **构建** | CMake 3.18+ / Makefile |
| **依赖** | cuBLAS, cuRAND, Google Test v1.14.0 (FetchContent) |
| **GPU** | Compute Capability 7.0+ (Volta → Hopper) |
| **质量** | clang-format, GitHub Actions CI |

## GPU 架构参考

| GPU 系列 | 架构 | Compute Capability | 构建参数 |
|----------|------|-------------------|---------|
| Tesla V100 | Volta | sm_70 | `GPU_ARCH=sm_70` |
| RTX 2080 | Turing | sm_75 | `GPU_ARCH=sm_75` |
| RTX 3090 / A100 | Ampere | sm_80 / sm_86 | `GPU_ARCH=sm_86` |
| RTX 4090 / L40 | Ada Lovelace | sm_89 | `GPU_ARCH=sm_89` |
| H100 | Hopper | sm_90 | `GPU_ARCH=sm_90` |

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — NVIDIA 官方
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) — Simon Boehm
- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA 高性能 GEMM 模板库
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/) — cuBLAS API 文档
- [Roofline Model](https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/) — 性能建模方法论

---

[View on GitHub](https://github.com/LessUp/sgemm-optimization) · [English README](README.md) · [中文 README](README.zh-CN.md)
