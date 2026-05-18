---
title: 参考文献地图
---

# 参考文献地图

本页是支撑本白皮书论点的外部资料结构化索引。每个条目按类型分类，并链接到其最直接支持的章节。

## 主要技术参考

### CUDA 与 GPU 架构

| 资料 | 建立了什么 | 相关章节 |
|---|---|---|
| [CUDA C++ 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | 内存层次结构、warp 执行模型、共享内存布局 | 架构、学院 |
| [CUDA 最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | 内存合并、占用率、bank 冲突消除 | 学院（kernel 页面） |
| [PTX ISA 参考](https://docs.nvidia.com/cuda/parallel-thread-execution/) | WMMA 指令语义、矩阵 fragment 布局 | Tensor Core 路径 |

### cuBLAS

| 资料 | 建立了什么 | 相关章节 |
|---|---|---|
| [cuBLAS 开发者指南](https://docs.nvidia.com/cuda/cublas/) | GEMM API、精度模式、leading-dimension 约定 | 验证（oracle 定义） |

### Tensor Core / WMMA

| 资料 | 建立了什么 | 相关章节 |
|---|---|---|
| [WMMA API 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma) | Fragment 类型、load/store/compute API | 学院（kernel-tensor-core）、架构（tensor-core-path） |
| [Volta 架构白皮书](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf) | 第一代 Tensor Core 吞吐模型 | 研究（演进）、性能模型 |

## 基础论文

| 论文 | 贡献 | 主要支持 |
|---|---|---|
| Goto & van de Geijn (2008) — [矩阵乘法高性能剖析](https://dl.acm.org/doi/10.1145/1356052.1356053) | CPU GEMM 分层分块理论 | Tiled kernel 设计、共享内存 staging 原理 |
| Lai & Seznec (2013) — [Fermi 和 Kepler GPU 上 SGEMM 的性能上限分析与优化](https://dl.acm.org/doi/10.1145/2464996.2465013) | GPU SGEMM 分块与占用率分析 | Tiled kernel、双缓冲动机 |
| Whaley & Dongarra (1998) — ATLAS | 块大小的自动调优 | 块大小敏感性的历史背景 |
| Markidis et al. (2018) — [NVIDIA Tensor Core 可编程性、性能与精度](https://ieeexplore.ieee.org/document/8425500) | WMMA 编程模型与混合精度行为 | Tensor Core 路径设计 |

## 相关开源实现

| 仓库 | 关系 | 说明 |
|---|---|---|
| [CUTLASS](https://github.com/NVIDIA/cutlass) | 权威生产级 GEMM kernel 库 | 本项目不声称与之竞争的天花板 |
| [tinygrad / BEAM SGEMM](https://github.com/tinygrad/tinygrad) | 社区 SGEMM 探索 | 不同的教育框架；适合用来对照 |
| [siboehm/CUDA-GEMM-Optimization](https://github.com/siboehm/CUDA-GEMM-Optimization) | 逐步讲解 SGEMM 的教程 | 教育结构上最直接可比的项目 |
| [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) | 中文 SGEMM 练习仓库 | 双语对照；不同的 kernel 演进顺序 |

## 如何使用本地图

参考文献地图不是论文末尾的参考书目，而是一个**活跃索引**，将白皮书中的每个论断与其支撑资料相连接。

如果你想质疑某个论断：
1. 找到白皮书中提出该论断的章节。
2. 在上表中找到对应的支撑资料。
3. 打开资料，检查该论断是否有适当的范围界定。

如果某个论断不在表中，它要么直接来源于实现本身（通过阅读代码可验证），要么是文本中明确标注为待决问题的开放性问题。

## 相关页面

- [参考资料清单](./references) — 完整的注释参考列表（含阅读说明）
- [论文索引](./papers) — 聚焦的学术阅读列表
- [相关项目](./related-projects) — 项目范围的对照背景
- [演进思考](./evolution) — 外部资料如何影响了当前设计
- [性能案例库](./performance-casebook) — 如何对照外部 benchmark 解读实测结果
