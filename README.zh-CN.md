# SGEMM Optimization: From Naive to Tensor Core

[English](README.md) | 简体中文

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)

从零手写极致优化的矩阵乘法 - HPC 领域的 "Hello World"

## 项目概述

本项目实现了一个从最简单的三层循环到接近 cuBLAS 性能的 CUDA SGEMM (Single-precision General Matrix Multiply) 优化演进过程。通过渐进式优化，展示 GPU 编程中的核心优化技术。

## 性能说明

实际 GFLOPS 会受到 GPU 型号、CUDA 版本和问题规模影响。
当前 benchmark 会报告两种 Tensor Core 口径：

- **Tensor Core (WMMA end-to-end)**：包含 FP32→FP16 转换，以及对非 WMMA 兼容尺寸的安全回退。
- **Tensor Core (WMMA compute-only)**：仅统计 WMMA 计算路径时间，只会在 `M/K/N` 都是 16 的倍数时显示。

项目中的验证容差已统一：

- 标准 FP32 kernel：`rtol=1e-3`, `atol=1e-4`
- Tensor Core 混合精度路径：`rtol=5e-2`, `atol=1e-2`

默认 benchmark 集合包含：

- 对齐方阵：`512x512x512`、`1024x1024x1024`
- 一组对齐非方阵：`256x384x640`
- 一组非对齐边界尺寸：`511x513x1025`，用于验证 Tensor Core 安全回退

> 注意：程序输出中的 theoretical peak 和 roofline 数据都是近似分析值，不是严格硬件峰值。

## 优化版本

| 版本 | 描述 | 关键技术 |
|------|------|----------|
| Naive | 基础三层循环 | 每线程计算一个输出元素 |
| Tiled | 共享内存分块 | 数据复用，减少全局内存访问 |
| Bank Conflict Free | 消除 bank 冲突 | 共享内存 padding (+1) |
| Double Buffer | 双缓冲流水线 | 通过重组 tile staging 降低访存停顿 |
| Tensor Core | WMMA API | 硬件加速矩阵运算 (FP16→FP32) |

## 构建与运行

### 环境要求

- CUDA Toolkit 11.0+
- cuBLAS (CUDA 自带)
- GPU: Volta (sm_70) 或更新架构
- Google Test (可选，用于属性测试)

### 编译与运行

推荐优先使用 CMake：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/bin/sgemm_benchmark
./build/bin/sgemm_benchmark --dims 256 384 640
./build/bin/sgemm_benchmark -a
```

Makefile 可用于快速本地构建：

```bash
make GPU_ARCH=sm_86
make benchmark
make test
make clean
```

### 输出说明

程序会根据输入尺寸和 GPU 能力输出不同 kernel 的结果。
当尺寸满足 WMMA 要求时，会同时显示：

- `Tensor Core (WMMA end-to-end)`：包含 FP32→FP16 转换与安全回退语义
- `Tensor Core (WMMA compute-only)`：仅统计 WMMA 计算路径

具体数值会随 GPU、CUDA 版本和尺寸而变化，因此 README 不再固定展示单一机器上的性能表。

## 目录结构

```
sgemm-optimization/
├── src/
│   ├── kernels/
│   │   ├── naive_sgemm.cuh           # Naive: 基础三层循环
│   │   ├── tiled_sgemm.cuh           # Tiled: 共享内存分块
│   │   ├── bank_conflict_free_sgemm.cuh  # 消除 bank 冲突
│   │   ├── double_buffer_sgemm.cuh   # 双缓冲流水线
│   │   └── tensor_core_sgemm.cuh     # Tensor Core (WMMA API)
│   ├── utils/
│   │   ├── cuda_utils.cuh            # CUDA 工具函数和错误检查
│   │   ├── benchmark.cuh             # 性能测试框架 (CUDA Events)
│   │   └── verify.cuh                # 正确性验证 (vs cuBLAS)
│   └── main.cu                       # 主程序入口
├── tests/
│   └── test_sgemm.cu                 # Google Test 属性测试
├── .kiro/specs/sgemm-optimization/   # 设计规范文档
│   ├── requirements.md               # 需求文档
│   ├── design.md                     # 设计文档
│   └── tasks.md                      # 实现任务清单
├── roofline_data_*.csv               # Roofline 分析数据
├── Makefile
└── README.md
```

## 核心优化技术详解

### 1. 内存合并访问 (Memory Coalescing)

**问题：** Naive 版本中，访问矩阵 B 的列是非合并的，导致带宽浪费。

```cpp
// ❌ 非合并访问 (Naive) - 不同线程访问不连续地址
float b = B[k * N + col];  // stride = N

// ✅ 合并访问 (Tiled 加载阶段) - 同一 warp 访问连续地址
Bs[ty][tx] = B[bRow * N + bCol];  // stride = 1
```

**效果：** 合并访问可将内存带宽利用率从 ~12.5% 提升到接近 100%。

### 2. 共享内存分块 (Tiling)

**原理：** 将大矩阵分成小块加载到共享内存，实现数据复用。

```cpp
// 每个 tile 的数据被复用 TILE_SIZE 次
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

// 数据复用: 每个元素被读取 1 次，使用 TILE_SIZE 次
for (int k = 0; k < TILE_SIZE; ++k) {
    sum += As[ty][k] * Bs[k][tx];
}
```

**复杂度分析：**
- 全局内存访问: O(N³/TILE_SIZE) → 减少 TILE_SIZE 倍
- 共享内存访问: O(N³) (但延迟低 ~100x)

### 3. Bank Conflict 消除

**问题：** 共享内存分为 32 个 bank，列访问时所有线程访问同一 bank。

```cpp
// ❌ 有 bank conflict - 32 路冲突，串行化
__shared__ float As[32][32];
float a = As[k][ty];  // 所有线程访问 bank[ty % 32]

// ✅ 无 bank conflict - padding 使列访问跨越不同 bank
__shared__ float As[32][33];  // +1 padding
float a = As[k][ty];  // 线程 i 访问 bank[(k*33 + ty) % 32]
```

**效果：** 消除 32 路 bank conflict，共享内存带宽提升 ~32x。

### 4. 双缓冲流水线 (Double Buffering)

**原理：** 使用两个缓冲区交替进行加载和计算，掩盖内存延迟。

```
Without Double Buffer:
  Load[0] → Compute[0] → Load[1] → Compute[1] → ...

With Double Buffer:
  Load[0] → Load[1] + Compute[0] → Load[2] + Compute[1] → ...
```

```cpp
__shared__ float As[2][TILE_SIZE][TILE_SIZE];  // 双缓冲
__shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

// 预加载第一个 tile
loadTile(As[0], Bs[0], 0);
__syncthreads();

for (int t = 0; t < numTiles; ++t) {
    int curr = t % 2;
    int next = (t + 1) % 2;
    
    // 异步预取下一个 tile
    if (t + 1 < numTiles) {
        loadTile(As[next], Bs[next], t + 1);
    }
    
    // 计算当前 tile
    computeTile(As[curr], Bs[curr]);
    __syncthreads();
}
```

### 5. Tensor Core (WMMA API)

当前实现中，Tensor Core 入口在不满足 `sm_70+` 或 `M/K/N` 非 16 对齐时会自动回退到稳定的 FP32 kernel，而不是继续执行不安全的 WMMA 访问。


**特点：**
- 专用矩阵计算单元，执行 D = A×B + C
- 支持 FP16 输入，FP32 累加 (混合精度)
- 一个 warp (32 线程) 协作执行 16×16×16 矩阵乘法
- 理论峰值远超 CUDA Core (~8x on Ampere)

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// 声明 fragment
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

// 加载数据 (FP32 → FP16 转换)
load_matrix_sync(a_frag, A_fp16, 16);
load_matrix_sync(b_frag, B_fp16, 16);

// 执行矩阵乘法
mma_sync(c_frag, a_frag, b_frag, c_frag);

// 存储结果
store_matrix_sync(C, c_frag, N, mem_row_major);
```

**注意：** Tensor Core 使用 FP16 中间精度，需要放宽验证容差 (rtol=5e-2, atol=1e-2)。

### 6. Roofline Model 分析

**算术强度 (Arithmetic Intensity):**
```
AI = FLOPs / Bytes
   = 2MNK / (4 × (MK + KN + MN))
   ≈ N/3  (对于方阵 M=N=K)
```

**分析方法：**
1. 计算 kernel 的算术强度
2. 在 Roofline 图上定位
3. 判断瓶颈：计算受限 or 内存受限
4. 针对性优化

**SGEMM 特点：**
- 小矩阵 (N<256): 内存受限 → 优化访存
- 大矩阵 (N>1024): 计算受限 → 优化计算

## 正确性验证

本项目使用 numpy 风格的 allclose 验证：

```cpp
// |test - ref| <= atol + rtol × |ref|
bool passed = abs_error <= atol + rtol * fabs(ref_val);
```

**容差设置：**
- 标准 kernel: rtol=1e-3, atol=1e-4
- Tensor Core: rtol=5e-2, atol=1e-2 (FP16 精度损失)

## 属性测试 (Property-Based Testing)

测试文件 `tests/test_sgemm.cu` 目前覆盖：

1. **标准 kernel 数值正确性** - 方阵与非方阵结果对齐 cuBLAS
2. **Tensor Core fast path** - 16 对齐尺寸下验证 WMMA 路径
3. **Tensor Core fallback** - 非对齐尺寸时安全回退到 FP32 kernel
4. **小尺寸与边界尺寸** - 包含 `1x1x1` 和非对齐 edge case
5. **Error Detection** - 验证工具与 benchmark 容差保持一致

运行测试：
```bash
cmake --build build --target test_sgemm
ctest --test-dir build

# 或
make test
```

## 面试要点总结

| 优化技术 | 解决的问题 | 性能提升 |
|----------|-----------|----------|
| Coalescing | 非合并访问浪费带宽 | ~8x |
| Tiling | 重复访问全局内存 | ~2-5x |
| Bank Conflict Free | 共享内存访问串行化 | ~1.1-1.3x |
| Double Buffer | 内存延迟暴露 | ~1.1-1.2x |
| Tensor Core | CUDA Core 计算瓶颈 | ~3-4x |

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [CUTLASS](https://github.com/NVIDIA/cutlass) - NVIDIA 高性能 GEMM 库
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
- [Roofline Model](https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/)

## License

MIT License
