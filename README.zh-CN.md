# SGEMM Optimization

[![CI](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/sgemm-optimization/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/sgemm-optimization/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)

[English](README.md) | 简体中文

这是一个面向学习与面试展示的 CUDA SGEMM 工程化项目：从可读的 FP32 baseline kernel 演进到带保护回退的 Tensor Core WMMA，并通过 cuBLAS 对照建立可信验证。

## 为什么它更有竞争力

- **优化链条完整**：naive -> tiled -> bank-conflict-free -> double-buffer -> Tensor Core。
- **证据优先表达**：性能结论与正确性策略、测量范围一起呈现。
- **接口保持一致**：FP32 kernel 使用统一 `(A, B, C, M, K, N, stream)` launcher 契约。
- **面试友好叙事**：架构、方法论、验证与参考资料共同支撑同一条公共叙事。
- **中英文镜像文档**：公开页面结构保持一致，便于传播与复用。

## 快速开始

```bash
git clone https://github.com/LessUp/sgemm-optimization.git
cd sgemm-optimization

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/bin/sgemm_benchmark -a
ctest --test-dir build
```

运行时测试和 benchmark 需要本地 CUDA GPU。托管 CI 只覆盖格式、仓库结构、OpenSpec / 治理，以及 Pages 构建检查。

## 推荐入口（GitHub Pages）

| 目标 | 入口 |
|------|------|
| 打开中文首页 | [中文首页](https://lessup.github.io/sgemm-optimization/zh/) |
| 打开英文首页 | [Docs Home](https://lessup.github.io/sgemm-optimization/en/) |
| 编译运行一次 | [快速上手](https://lessup.github.io/sgemm-optimization/zh/getting-started) |
| 了解项目差异化 | [架构概述](https://lessup.github.io/sgemm-optimization/zh/architecture/) |
| 准备面试表达 | [方法论](https://lessup.github.io/sgemm-optimization/zh/methodology/) |
| 查看可信边界 | [验证概览](https://lessup.github.io/sgemm-optimization/zh/validation/) |
| 追溯技术来源 | [参考文献](https://lessup.github.io/sgemm-optimization/zh/references) |
| 阅读规范源 | [OpenSpec 规范](openspec/specs/) |

## 验证边界

| 环境 | 可以信任什么 |
|------|--------------|
| 托管 CI | 格式、文档/结构检查、OpenSpec 治理、Pages 可构建性 |
| 本地 CUDA GPU | 运行时正确性与 benchmark 性能 |

这种拆分是刻意设计：CI 负责仓库健康，真实 GPU 负责运行时与性能结论。

## 源码地图

```text
src/kernels/   CUDA SGEMM kernel 实现
src/utils/     CUDA RAII、验证与 benchmark 工具
src/main.cu    benchmark CLI
tests/         基于 cuBLAS 的 Google Test 覆盖
docs/          中英文 Pages 文档（含 /en 与 /zh）
openspec/      稳定 specs 与变更工作流
```

## 许可证

MIT，见 [LICENSE.md](LICENSE.md)。
