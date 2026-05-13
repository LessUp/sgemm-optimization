---
title: 架构概览
---

# 架构概览

仓库包含什么，验证责任在哪里



## 仓库表面

| 表面 | 角色 |
|------|------|
| `README.md` | 仓库入口和快速开始 |
| `index.md` + `docs/` | 公开首页和学习型文档 |
| `openspec/specs/` | 稳定权威需求和治理 |
| `openspec/changes/` | 活动实现计划和 delta specs |
| `.github/workflows/` | CI 安全验证和 Pages 部署 |

仓库有意将**公开解释**与**规范流程**分离。OpenSpec 治理；文档教学；README 介绍。



## 验证边界

| 领域 | 本地 GPU 机器 | 托管 CI |
|------|---------------|---------|
| CUDA 编译 | 是 | 是 |
| 运行时正确性 | 是 | 否 |
| Benchmark | 是 | 否 |
| OpenSpec/仓库检查 | 是 | 是 |
| GitHub Pages 构建 | 可选 | 是 |

这种划分是刻意的。仓库不假装 CI 能替代真正的 CUDA 运行时环境。



## 相关参考

- [学习路径](/zh/learning-path)
- [快速上手](/zh/getting-started)
- [规范索引](https://github.com/LessUp/sgemm-optimization/tree/master/openspec/specs/)
- [稳定架构规范](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/specs/architecture/spec.md)
