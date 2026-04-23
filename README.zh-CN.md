# SGEMM Optimization

[![CI](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/sgemm-optimization/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/sgemm-optimization/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)

[English](README.md) | 简体中文

这是一个渐进式 CUDA SGEMM 教程与参考实现仓库。项目用 5 个手写 kernel 展示从最朴素矩阵乘法到 Tensor Core WMMA 的优化路径，并配套 cuBLAS 对标验证、benchmark 工具，以及基于 OpenSpec 的仓库治理流程。

## 为什么值得看

- **优化链条完整**：naive -> tiled -> bank-conflict-free -> double-buffer -> Tensor Core
- **代码可读性强**：每一级优化都独立成文件，并保持统一的 launch 接口
- **验证链路完整**：用 cuBLAS 做正确性对照，并区分标准 FP32 与 Tensor Core 容差
- **工程边界清晰**：仓库通过 OpenSpec 管理规范、文档、workflow 和收尾整理

## Kernel 演进

| 阶段 | 文件 | 核心思路 |
|------|------|----------|
| Naive | `src/kernels/naive_sgemm.cuh` | 基础三层循环映射 |
| Tiled | `src/kernels/tiled_sgemm.cuh` | 共享内存分块 |
| Bank-Free | `src/kernels/bank_conflict_free_sgemm.cuh` | `[TILE_SIZE][TILE_SIZE+1]` padding 消除 bank 冲突 |
| Double Buffer | `src/kernels/double_buffer_sgemm.cuh` | tile staging 重叠与延迟隐藏 |
| Tensor Core | `src/kernels/tensor_core_sgemm.cuh` | WMMA 路径与安全 FP32 回退 |

## 快速开始

```bash
git clone https://github.com/LessUp/sgemm-optimization.git
cd sgemm-optimization

# 推荐：CMake
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/bin/sgemm_benchmark -a
ctest --test-dir build
```

```bash
# 本地快速方案
make GPU_ARCH=sm_86
make benchmark
make test
```

## 验证边界

- **本地 GPU 机器**：运行时测试、正确性验证、benchmark
- **GitHub Actions**：格式/风格、CUDA 编译验证、OpenSpec/仓库结构校验、Pages 部署

标准 FP32 kernel 使用 `rtol=1e-3`、`atol=1e-4`；Tensor Core 路径使用 `rtol=5e-2`、`atol=1e-2`。

## 建议继续阅读

- [快速上手](docs/getting-started.md)
- [学习路径](docs/learning-path.md)
- [架构概览](docs/architecture.md)
- [Benchmark 说明](docs/benchmark-results.md)
- [规范索引](specs.md)
- [GitHub Pages 站点](https://lessup.github.io/sgemm-optimization/)

## 仓库结构

```text
src/
├── kernels/        # 五个 SGEMM kernel 版本
├── utils/          # CUDA RAII、验证和 benchmark 工具
└── main.cu         # Benchmark 入口
tests/
└── test_sgemm.cu   # Google Test 测试套件
docs/               # 面向读者的学习型文档
openspec/           # 稳定 specs、变更和流程说明
```

## 开发流程

涉及仓库结构、规范或文档的非小改动，默认遵循：

1. `/opsx:explore`
2. `/opsx:propose "description"`
3. `/opsx:apply`
4. `/review`
5. `/opsx:archive`

稳定规范位于 `openspec/specs/`，活动实现计划位于 `openspec/changes/<change>/`。

## 许可证

MIT，见 [LICENSE](LICENSE)。
