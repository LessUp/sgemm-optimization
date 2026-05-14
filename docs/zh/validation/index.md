---
title: 验证
---

# 验证

本章节解释**为什么仓库里的性能结论值得信任，以及这种信任到哪里为止**。

方法论解释“如何优化”。验证解释这些优化结论周围的证据边界：正确性阈值、benchmark 范围标注、托管 CI 的限制，以及可复现性的要求。

## 验证模型

| 证据表面 | 能证明什么 | 不能证明什么 |
|----------|------------|--------------|
| 托管 CI + 本地结构检查 | 文档/规范结构、Pages 可发布性、格式化/治理流程检查，以及仓库健康检查 | GPU 运行时正确性、CUDA benchmark 数字、硬件相关提速 |
| 本地 GPU 机器上的 `ctest --test-dir build` | 以项目 cuBLAS oracle 为基准的运行时正确性 | 通用性能结论 |
| 本地 benchmark 执行 | 某个 GPU、某条命令、某个范围标签下的性能行为 | 其他 GPU、其他 CUDA 栈或未标注工作负载上的结果 |

## 规范验证页面

| 需求 | 页面 |
|------|------|
| 理解正确性阈值与 oracle 策略 | [正确性策略](/zh/validation/correctness-policy) |
| 解释 benchmark 标签与公开数字 | [Benchmark 范围](/zh/validation/benchmark-scope) |
| 负责任地复现结果 | [可复现性](/zh/validation/reproducibility) |
| 查看代表性 benchmark 快照 | [Benchmark 结果](/zh/benchmark-results) |

## 托管 CI 能证明什么

托管 CI 被信任来证明仓库健康：文档结构、Pages 构建能力、格式化检查，以及 OpenSpec / 治理一致性。它保证公共表面保持连贯。

托管 CI **不** 被用来证明 CUDA 运行时行为或 benchmark 性能。这些结论必须来自真实 GPU 机器。

## 只有本地 GPU 运行才能证明什么

以下内容必须依赖本地 GPU 执行：

- 基于 cuBLAS 的运行时正确性检查
- Tensor Core 快路径与 fallback 行为
- benchmark 数字，包括端到端与 compute-only 的差异
- 与占用率、staging、内存行为相关的架构级结论

## 如何阅读公开数字

把仓库中的每个数字都看成**有范围边界的证据**，而不是通用承诺。

- 先看 GPU 型号与 CUDA 环境
- 再看 benchmark 标签
- 再看 shape 集合
- 最后才与其他结果对比

如果缺少其中任意一项，这个数字更像提示，而不是结论。
