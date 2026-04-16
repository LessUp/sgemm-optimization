---
layout: default
title: SGEMM Optimization
---

# SGEMM Optimization: From Naive to Tensor Core

[![CI](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/LessUp/sgemm-optimization/blob/main/LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)

**从零手写极致优化的 CUDA 矩阵乘法 — HPC 领域的 "Hello World"**

五个渐进式优化的 kernel 变体，展示 GPU 核心优化技术，从最朴素的三层循环到 **Tensor Core WMMA**。

---

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/LessUp/sgemm-optimization.git
cd sgemm-optimization

# CMake 构建 (推荐)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 运行基准测试
./build/bin/sgemm_benchmark -a
```

---

## 📊 性能总览

在 NVIDIA RTX 3060 Laptop GPU (Ampere, sm_86) 上实测，矩阵维度 1024×1024×1024：

| Kernel | GFLOPS | vs cuBLAS | 耗时 | 核心技术 |
|--------|-------:|----------:|------:|----------|
| **cuBLAS** | 5727 | 100% | 0.375 ms | NVIDIA 优化库 |
| **Tensor Core** | 2300 | 40.2% | 0.934 ms | WMMA, FP16→FP32 |
| **Tiled** | 753 | 13.1% | 2.853 ms | 共享内存分块 |
| **Double Buffer** | 701 | 12.2% | 3.064 ms | 双缓冲流水线 |
| **Bank-Free** | 673 | 11.8% | 3.190 ms | Bank conflict 消除 |
| **Naive** | 604 | 10.6% | 3.553 ms | 基准实现 |

---

## 🔄 优化演进路线

```
┌─────────┐     ┌──────────┐     ┌──────────────┐     ┌───────────────┐
│  Naive  │────▶│  Tiled   │────▶│  Bank-Free   │────▶│ Double Buffer │
└─────────┘     └──────────┘     └──────────────┘     └───────┬───────┘
                                                          │
                                                          ▼
                                              ┌───────────────────┐
                                              │   Tensor Core     │
                                              │   (WMMA API)      │
                                              └───────────────────┘
```

| 阶段 | 变更内容 | 性能提升 |
|------|---------|---------|
| Naive → Tiled | 共享内存分块 | 数据复用 ↑ |
| Tiled → Bank-Free | Padding 消除冲突 | 带宽 ↑ |
| Bank-Free → Double Buffer | 计算访存重叠 | 延迟隐藏 |
| → Tensor Core | WMMA API | ~8× 峰值 |

---

## 🛠️ 核心技术

### 1. 内存合并访问 (Coalescing)
确保同一 warp 内线程访问连续地址，带宽利用率从 ~12.5% 提升至 ~100%。

### 2. 共享内存分块 (Tiling)
将大矩阵分块加载到共享内存，全局内存访问减少 TILE_SIZE 倍。

### 3. Bank Conflict 消除
通过 `[32][33]` padding 消除 32 路 bank conflict，共享内存带宽恢复。

### 4. 双缓冲流水线 (Double Buffering)
两个缓冲区交替加载和计算，掩盖全局内存延迟。

### 5. Tensor Core (WMMA)
使用 WMMA API 调用专用矩阵单元，FP16 输入 + FP32 累加。

---

## 📁 项目结构

```
sgemm-optimization/
├── src/
│   ├── kernels/           # 5 个优化 kernel
│   └── utils/             # 工具函数
├── tests/                 # Google Test
├── CHANGELOG.md           # 版本历史
├── CMakeLists.txt         # CMake 构建
└── Makefile               # Make 构建
```

---

## 🧪 测试与验证

```bash
# 运行测试
cmake --build build --target test_sgemm
ctest --test-dir build
```

| 属性 | 验证内容 |
|------|---------|
| 数值正确性 | 与 cuBLAS 结果一致 |
| Tensor Core | 16 对齐尺寸验证 |
| Fallback | 非对齐尺寸安全回退 |

---

## 📖 GPU 架构支持

| GPU | 架构 | 构建参数 |
|-----|------|---------|
| V100 | Volta | `GPU_ARCH=sm_70` |
| RTX 3090 / A100 | Ampere | `GPU_ARCH=sm_86` |
| RTX 4090 | Ada | `GPU_ARCH=sm_89` |
| H100 | Hopper | `GPU_ARCH=sm_90` |

---

## 📚 参考资料

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)

---

<div style="text-align: center; margin-top: 2rem;">
  <a href="https://github.com/LessUp/sgemm-optimization">🔍 View on GitHub</a> ·
  <a href="README.md">English README</a> ·
  <a href="README.zh-CN.md">中文 README</a> ·
  <a href="CHANGELOG.md">📜 Changelog</a>
</div>
