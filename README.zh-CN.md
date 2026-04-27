# SGEMM Optimization

[![CI](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/sgemm-optimization/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/sgemm-optimization/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)

[English](README.md) | 简体中文

渐进式 CUDA SGEMM 教程与参考实现，从朴素 kernel 到 Tensor Core WMMA。包含 cuBLAS 对标验证、benchmark 工具，以及 OpenSpec 仓库治理流程。

## 为什么值得看

- **优化链条完整**：naive → tiled → bank-conflict-free → double-buffer → Tensor Core WMMA
- **代码可读性强**：每级优化独立成文件，保持统一的 launch 接口
- **验证链路完整**：cuBLAS 正确性对照，区分标准 FP32 与 Tensor Core 容差
- **工程边界清晰**：OpenSpec 管理规范、文档、workflow 和收尾整理

## 优化阶梯

| 阶段 | Kernel | 学习重点 |
|-----:|--------|----------|
| 1 | [Naive](zh/docs/kernel-naive) | 线程到输出的映射与基线代价 |
| 2 | [Tiled](zh/docs/kernel-tiled) | 共享内存分块与数据复用 |
| 3 | [Bank-Free](zh/docs/kernel-bank-free) | Padding 消除 32 路 bank 冲突 |
| 4 | [Double Buffer](zh/docs/kernel-double-buffer) | 分阶段 tile 加载与延迟隐藏 |
| 5 | [Tensor Core](zh/docs/kernel-tensor-core) | WMMA 使用与安全的 FP32 回退 |

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

## 从哪里开始

| 如果你想... | 从这里开始 |
|-------------|-----------|
| 编译运行一次 | [快速上手](zh/docs/getting-started) |
| 学习优化路径 | [学习路径](zh/docs/learning-path) |
| 了解仓库结构 | [架构概览](zh/docs/architecture) |
| 查看性能数据 | [Benchmark 结果](zh/docs/benchmark-results) |
| 查看治理规则 | [规范索引](zh/specs) |

## 验证边界

- **本地 GPU 机器**：运行时测试、正确性验证、benchmark
- **GitHub Actions**：格式/风格、CUDA 编译、OpenSpec 校验、Pages 部署

标准 FP32 kernel：`rtol=1e-3`、`atol=1e-4`。Tensor Core 路径：`rtol=5e-2`、`atol=1e-2`。

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

## 项目状态

本仓库处于**归档就绪**状态。所有 kernel 实现已完成，测试通过，文档对齐。非小改动遵循 OpenSpec 工作流：

1. `/opsx:explore` — 明确范围与权衡
2. `/opsx:propose "描述"` — 创建变更文档
3. `/opsx:apply` — 实施任务
4. `/review` — 质量门禁
5. `/opsx:archive` — 合并并关闭

稳定规范：`openspec/specs/`。活动变更：`openspec/changes/<change>/`。

## 许可证

MIT，见 [LICENSE](LICENSE)。
