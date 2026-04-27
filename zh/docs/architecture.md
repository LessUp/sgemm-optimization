---
layout: default
title: 架构概览
nav_order: 8
permalink: /zh/docs/architecture
lang: zh-CN
page_key: zh-architecture
lang_ref: architecture
---

# 架构概览
{: .fs-8 }

仓库包含什么，验证责任在哪里
{: .fs-6 .fw-300 }

---

## 系统形态

```text
main.cu
  ├── benchmark 编排
  ├── 验证流程
  └── CLI 参数处理

src/kernels/
  ├── naive
  ├── tiled
  ├── bank-conflict-free
  ├── double-buffer
  └── tensor-core

src/utils/
  ├── CUDA RAII 和错误处理
  ├── benchmark 辅助函数
  └── 验证辅助函数

tests/
  └── Google Test cuBLAS 对标覆盖
```

---

## 仓库表面

| 表面 | 角色 |
|------|------|
| `README.md` | 仓库入口和快速开始 |
| `index.md` + `docs/` | 公开首页和学习型文档 |
| `openspec/specs/` | 稳定权威需求和治理 |
| `openspec/changes/` | 活动实现计划和 delta specs |
| `.github/workflows/` | CI 安全验证和 Pages 部署 |

仓库有意将**公开解释**与**规范流程**分离。OpenSpec 治理；文档教学；README 介绍。

---

## Kernel 契约

所有 kernel launcher 遵循相同形态：

```cpp
template<int TILE_SIZE = 32>
void launch_xxx_sgemm(
    const float* A, const float* B, float* C,
    int M, int K, int N,
    cudaStream_t stream = 0
);
```

共享的 launcher 契约使得 benchmark、替换和验证 kernel 变得简单，无需修改外围框架。

---

## 验证边界

| 领域 | 本地 GPU 机器 | 托管 CI |
|------|---------------|---------|
| CUDA 编译 | 是 | 是 |
| 运行时正确性 | 是 | 否 |
| Benchmark | 是 | 否 |
| OpenSpec/仓库检查 | 是 | 是 |
| GitHub Pages 构建 | 可选 | 是 |

这种划分是刻意的。仓库不假装 CI 能替代真正的 CUDA 运行时环境。

---

## 仓库级设计选择

1. **渐进式 kernel** 保持优化步骤可读。
2. **RAII 封装和异常式错误传播** 让 CUDA 资源处理可预测。
3. **OpenSpec 治理仓库级变更** 使文档、workflow 和验证保持对齐。
4. **文档保持角色分工**：README 介绍，Pages 教学，OpenSpec 定义规则。

---

## 相关参考

- [学习路径](learning-path)
- [快速上手](getting-started)
- [规范索引](../../specs)
- [稳定架构规范](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/specs/architecture/spec.md)
