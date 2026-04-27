---
layout: default
title: 快速上手
parent: 首页
nav_order: 1
permalink: /zh/docs/getting-started/
lang: zh-CN
page_key: zh-getting-started
lang_ref: getting-started
---

# 快速上手
{: .fs-8 }

编译、运行和验证项目，无需猜测工具链
{: .fs-6 .fw-300 }

---

## 环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA Volta (`sm_70`) 或更新 |
| CUDA Toolkit | 11.0+ |
| CMake | 3.18+ |
| 主机编译器 | GCC 9+ 或 Clang 10+ |

Tensor Core benchmark 需要 `sm_70+` 和 16 对齐的维度。不满足条件时代码仍可在受保护的 FP32 路径运行。

---

## 推荐编译流程

```bash
git clone https://github.com/LessUp/sgemm-optimization.git
cd sgemm-optimization

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

运行默认 benchmark：

```bash
./build/bin/sgemm_benchmark
```

运行完整 benchmark 集：

```bash
./build/bin/sgemm_benchmark -a
```

运行测试：

```bash
ctest --test-dir build
```

---

## 选择 CUDA 架构

默认情况下：

- CMake 3.24+ 使用 `native`
- 较旧 CMake 回退到仓库的显式架构列表

如需覆盖，使用 CMake 原生变量：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
```

本地 Makefile 快速流程：

```bash
make GPU_ARCH=sm_86
make benchmark
make test
```

---

## 验证边界

| 环境 | 运行什么 |
|------|----------|
| 本地 GPU 机器 | benchmark、运行时验证、`ctest` |
| 托管 CI | 格式化、编译验证、OpenSpec/仓库检查、Pages |

这种划分是刻意的：GitHub 托管 runner 验证仓库健康，而性能和 CUDA 运行时正确性仍需真正的 GPU 机器。

---

## 常用命令

```bash
# 单个显式 benchmark 用例
./build/bin/sgemm_benchmark --dims 256 384 640

# 更长的 benchmark 运行
./build/bin/sgemm_benchmark -a --warmup 10 --benchmark 50

# OpenSpec 验证
openspec validate --all
```

---

## 接下来去哪

- [学习路径](learning-path/)
- [架构概览](architecture/)
- [Benchmark 结果](benchmark-results/)
