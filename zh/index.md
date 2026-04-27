---
layout: default
title: 首页
nav_order: 1
has_children: true
permalink: /zh/
description: 双语 CUDA SGEMM 优化教程与参考实现，从朴素 kernel 到 Tensor Core WMMA。
lang: zh-CN
page_key: zh-home
lang_ref: home
---

{: .hero-section }
# SGEMM Optimization
{: .hero-title }

从可读基线代码到 Tensor Core WMMA
{: .hero-subtitle }

[🚀 快速上手](zh/docs/getting-started){: .btn .fs-5 .mb-4 .mb-md-0 }
[📚 学习路径](zh/docs/learning-path){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }

---

## 这个项目有什么用

本仓库是一个紧凑的 CUDA GEMM 学习与参考项目：

- **渐进式**：五个 kernel 变体展示每步优化的变化
- **可验证**：每个 kernel 都与 cuBLAS 对标检查
- **实用性**：benchmark 和测试入口已预置
- **可维护**：通过 OpenSpec 文档化仓库规则、workflow 和验证

---

## 优化阶梯

| 阶段 | Kernel | 学习重点 |
|-----:|--------|----------|
| 1 | [Naive](zh/docs/kernel-naive) | 线程到输出的映射与基线代价 |
| 2 | [Tiled](zh/docs/kernel-tiled) | 共享内存分块与数据复用 |
| 3 | [Bank-Free](zh/docs/kernel-bank-free) | Padding 消除 32 路 bank 冲突 |
| 4 | [Double Buffer](zh/docs/kernel-double-buffer) | 分阶段 tile 加载与延迟隐藏 |
| 5 | [Tensor Core](zh/docs/kernel-tensor-core) | WMMA 使用与安全的 FP32 回退 |

---

## 仓库内容

| 目录 | 用途 |
|------|------|
| `src/` | CUDA kernels、benchmark 入口、工具函数 |
| `tests/` | Google Test cuBLAS 对标验证 |
| `docs/` | 学习型技术文档（英文） |
| `zh/docs/` | 学习型技术文档（中文） |
| `openspec/` | 稳定规范、变更历史、流程说明 |

---

## 如何使用

### 1. 编译运行

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/bin/sgemm_benchmark -a
ctest --test-dir build
```

### 2. 跟随学习路线

- [快速上手](zh/docs/getting-started)
- [学习路径](zh/docs/learning-path)
- [架构概览](zh/docs/architecture)
- [Benchmark 说明](zh/docs/benchmark-results)

### 3. 查看项目规则

- [规范索引](zh/specs)
- [OpenSpec Workflow 说明](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/README.md)
- [仓库指南](https://github.com/LessUp/sgemm-optimization/blob/master/AGENTS.md)

---

## 验证边界

- **本地 GPU 机器**：运行时验证和 benchmark
- **托管 CI**：格式、编译、OpenSpec/仓库检查、Pages 部署

这种划分保证仓库诚实，不假装 GitHub 托管 runner 能替代真正的 CUDA 运行时环境。

---

## 继续探索

[📘 Benchmark 结果](zh/docs/benchmark-results){: .btn .btn-outline .mr-2 }
[🏗️ 架构概览](zh/docs/architecture){: .btn .btn-outline .mr-2 }
[⭐ 在 GitHub 查看](https://github.com/LessUp/sgemm-optimization){: .btn .btn-outline }
