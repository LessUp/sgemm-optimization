# SGEMM Optimization

[![CI](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/sgemm-optimization/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/sgemm-optimization/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)

[English](README.md) | 简体中文

这是一个被包装成技术白皮书和 Kernel 学院的 CUDA SGEMM 案例仓库。它从可读的 FP32 基线出发，沿着 tiled、bank-conflict-aware、double-buffer、带保护的 Tensor Core WMMA 路径逐级推进，并且把每一个性能结论都放回明确的验证边界里解释。

## 为什么它更强

- **优化阶梯可讲清楚**：每一级 kernel 都对应一次明确的瓶颈转移。
- **公共叙事以证据为先**：正确性策略、benchmark 范围和本地 GPU / 托管 CI 的信任边界始终跟着结论走。
- **面试表达友好**：Pages 站点被写成可解释、可答辩、可审查的技术叙事。
- **中英镜像完整**：英文与中文公共路由在整站范围内保持结构一致。

## 快速开始

```bash
git clone https://github.com/LessUp/sgemm-optimization.git
cd sgemm-optimization

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/bin/sgemm_benchmark -a
ctest --test-dir build
```

运行时测试和 benchmark 需要本地 CUDA GPU。托管 CI 负责验证格式检查、CUDA 编译、文档站点检查、路由完整性，以及 Pages 可构建性。

## GitHub Pages 入口

README 是执行摘要，长篇技术叙事在 GitHub Pages 上。

| 目标 | 入口 |
|------|------|
| 打开英文首页 | [English Home](https://lessup.github.io/sgemm-optimization/en/) |
| 打开中文首页 | [中文首页](https://lessup.github.io/sgemm-optimization/zh/) |
| 快速建立全局认知 | [项目导读](https://lessup.github.io/sgemm-optimization/zh/overview/) |
| 查看系统结构 | [架构](https://lessup.github.io/sgemm-optimization/zh/architecture/) |
| 系统学习 kernel 阶梯 | [学院](https://lessup.github.io/sgemm-optimization/zh/academy/) |
| 核对证据到底证明什么 | [验证](https://lessup.github.io/sgemm-optimization/zh/validation/) |
| 追溯论文和相关仓库 | [研究资料台](https://lessup.github.io/sgemm-optimization/zh/research/) |
| 查看贡献流程与验证命令 | [CONTRIBUTING.md](CONTRIBUTING.md) |

## 验证边界

| 环境 | 能证明什么 |
|------|------------|
| 托管 CI | 格式检查、CUDA 编译、文档结构、路由完整性、Pages 可构建性 |
| 本地 CUDA GPU | 运行时正确性、fallback 行为、benchmark 性能 |

这种拆分是刻意设计。CI 负责让仓库表面保持连贯，只有本地 CUDA 环境才能验证可执行文件、运行时行为和速度结论。

## 源码地图

```text
src/kernels/   CUDA SGEMM kernel 实现
src/utils/     CUDA RAII、验证与 benchmark 工具
src/main.cu    benchmark CLI
tests/         基于 cuBLAS 的 Google Test 覆盖
docs/          VitePress 白皮书与学院，公开镜像位于 /en 和 /zh
```

## 许可证

MIT，见 [LICENSE.md](LICENSE.md)。
