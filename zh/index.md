---
layout: default
title: 首页
nav_order: 1
has_children: true
permalink: /zh/
description: CUDA SGEMM 优化项目门户，用一条清晰路径讲透从 baseline kernel 到 Tensor Core WMMA 的演进。
lang: zh-CN
page_key: zh-home
lang_ref: home
---

{: .hero-section }
# SGEMM Optimization
{: .hero-title }

沿着一条可验证的 CUDA 矩阵乘法路线，从“一线程算一个输出”的 baseline 走到受保护的 Tensor Core WMMA。
{: .hero-subtitle }

[开始上手](zh/docs/getting-started/){: .btn .fs-5 .mb-4 .mb-md-0 }
[跟随优化阶梯](zh/docs/learning-path/){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }
[查看 GitHub](https://github.com/LessUp/sgemm-optimization){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }

---

## 这个项目解决什么问题

很多 GEMM 示例要么把细节藏在生产级库里，要么停留在玩具 kernel。本项目保留中间层：每一步优化都独立、可读、可 benchmark，并用 cuBLAS 做正确性对照。

<div class="perf-grid">
  <div class="perf-card">
    <div class="perf-label">路径</div>
    <div class="perf-value">5 阶段</div>
    <div class="perf-vs">baseline 到 WMMA</div>
  </div>
  <div class="perf-card">
    <div class="perf-label">参考</div>
    <div class="perf-value">cuBLAS</div>
    <div class="perf-vs">正确性对照</div>
  </div>
  <div class="perf-card">
    <div class="perf-label">接口</div>
    <div class="perf-value">1 形态</div>
    <div class="perf-vs">kernel 可直接替换</div>
  </div>
</div>

---

## 优化阶梯

| 阶段 | Kernel | 它回答的问题 |
|-----:|--------|--------------|
| 1 | [Naive](zh/docs/kernel-naive/) | 最简单且正确的 GPU 映射是什么？ |
| 2 | [Tiled](zh/docs/kernel-tiled/) | 共享内存复用会如何改变成本模型？ |
| 3 | [Bank-Free](zh/docs/kernel-bank-free/) | 分块之后，为什么 shared memory 布局仍然重要？ |
| 4 | [Double Buffer](zh/docs/kernel-double-buffer/) | 如何用 staged tiles 隐藏全局内存延迟？ |
| 5 | [Tensor Core](zh/docs/kernel-tensor-core/) | WMMA 什么时候有价值，包装器何时应该回退？ |

---

## 可信度来自哪里

| 关注点 | 项目做法 |
|--------|----------|
| 数值正确性 | Google Test 将各 kernel 与 cuBLAS 对照，区分 FP32 与混合精度容差。 |
| Benchmark 诚实性 | 输出中分开呈现 cuBLAS、FP32 kernel、Tensor Core 端到端、compute-only WMMA。 |
| 不支持的 shape | 公开 Tensor Core wrapper 对非 16 对齐维度使用受保护的 fallback。 |
| 托管 CI 边界 | CI 验证格式、CUDA 编译、OpenSpec 结构和 Pages；运行时测试仍属于本地 GPU 工作。 |

---

## 选择你的路线

| 如果你想... | 从这里开始 |
|-------------|-----------|
| 编译运行一次 | [快速上手](zh/docs/getting-started/) |
| 按设计顺序学习 | [学习路径](zh/docs/learning-path/) |
| 理解文件边界 | [架构概览](zh/docs/architecture/) |
| 解读 benchmark 输出 | [Benchmark 结果](zh/docs/benchmark-results/) |
| 查看稳定需求 | [规范索引](zh/specs/) |

---

## 仓库地图

```text
src/kernels/   五个 SGEMM kernel 实现
src/utils/     CUDA RAII、验证与 benchmark 工具
tests/         基于 cuBLAS 的 Google Test 覆盖
docs/          英文学习路径
zh/docs/       中文学习路径
openspec/      稳定 specs 与变更工作流
```

---

## 继续探索

[快速上手](zh/docs/getting-started/){: .btn .mr-2 }
[学习路径](zh/docs/learning-path/){: .btn .btn-outline .mr-2 }
[English home](../){: .btn .btn-outline }
