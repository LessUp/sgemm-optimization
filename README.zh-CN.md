# SGEMM Optimization

[![CI](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/sgemm-optimization/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/sgemm-optimization/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)

[English](README.md) | 简体中文

这是一个紧凑的 CUDA SGEMM 学习项目：从最容易读懂的 baseline kernel 出发，逐步推进到 Tensor Core WMMA，并用 cuBLAS 做正确性对照。

## 为什么值得看

- **优化链条完整**：naive -> tiled -> bank-conflict-free -> double-buffer -> Tensor Core。
- **接口保持一致**：FP32 kernel 都使用统一的 `(A, B, C, M, K, N, stream)` launcher 形态。
- **验证先行**：所有 kernel 与 cuBLAS 对照，FP32 与 Tensor Core 使用不同容差。
- **文档职责清晰**：README 只做仓库入口，完整学习路线放在 GitHub Pages。

## 快速开始

```bash
git clone https://github.com/LessUp/sgemm-optimization.git
cd sgemm-optimization

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/bin/sgemm_benchmark -a
ctest --test-dir build
```

运行时测试和 benchmark 需要本地 CUDA GPU。托管 CI 只覆盖编译、格式、仓库结构、OpenSpec 与 Pages 构建检查。

## 从哪里开始

| 目标 | 入口 |
|------|------|
| 打开项目站点 | [GitHub Pages](https://lessup.github.io/sgemm-optimization/zh/) |
| 编译运行一次 | [快速上手](zh/docs/getting-started.md) |
| 跟随优化路线 | [学习路径](zh/docs/learning-path.md) |
| 查看源码结构 | [架构概览](zh/docs/architecture.md) |
| 阅读稳定规范 | [规范索引](zh/specs.md) |

## 源码地图

```text
src/kernels/   CUDA SGEMM kernel 实现
src/utils/     CUDA RAII、验证与 benchmark 工具
src/main.cu    benchmark CLI
tests/         基于 cuBLAS 的 Google Test 覆盖
docs/          英文学习文档与 Pages 内容
zh/docs/       中文学习文档与 Pages 内容
openspec/      稳定 specs 与变更工作流
```

## 许可证

MIT，见 [LICENSE.md](LICENSE.md)。
