---
title: 延伸阅读路线
---

# 延伸阅读路线

这页不是“再给你一堆链接”，而是刻意回答：看完本站之后，下一步到底该往哪里学、为什么先学那个。

## GEMM 分块与层级化思维

当你已经能机械地看懂 tiled SGEMM，但还说不清它为什么成立时，走这条路线。

- 先回看 [分块内核](/zh/academy/kernel-tiled) 与 [Memory Flow](/zh/architecture/memory-flow)，把本仓库的具体例子放回脑中。
- 再读 [Anatomy of High-Performance Matrix Multiplication](/zh/research/references#论文与性能心智模型)，理解 blocking 与数据复用为什么是系统级设计问题。
- 最后对照 [CUTLASS](/zh/research/references#示范仓库与生产级样例)，观察生产级 CUDA 库如何表达分块层级与任务分工。

建议一直追问：

- 每一级 tile 到底在保护哪一层内存？
- 哪个设计改变是在降低带宽压力，哪个是在换取更高吞吐？
- 教学型 kernel 与生产模板库的边界差在哪里？

## 把 occupancy 当约束，不当口号

当你听到很多人说“occupancy 不够高”，但不知道它究竟是不是根因时，走这条路线。

- 先看 [双缓冲](/zh/academy/kernel-double-buffer) 与 [消除 Bank Conflict](/zh/academy/kernel-bank-free)，因为寄存器压力与 shared memory 布局通常就是 occupancy 变化的来源。
- 再配合 [CUDA Occupancy Calculator 与 Nsight Compute 资料](/zh/research/references#profiler、工具与诊断资料)，把 launch 配置和 active warps、寄存器、shared memory 限制对应起来。
- 然后回到 [诊断闭环](/zh/academy/diagnosis-loop)，把 occupancy 变成一个待验证假设，而不是万能答案。

建议一直追问：

- occupancy 下降，是 kernel 变差了，还是每个 block 做了更多有价值的工作？
- 先卡住你的到底是寄存器、shared memory，还是 block 配置？
- 哪个 profiler 指标可以推翻你当前的解释？

## 面向 SGEMM 的 Roofline 思维

当你想更严谨地区分“内存受限”和“计算受限”，而不是只凭直觉下结论时，走这条路线。

- 先用 [Benchmark 纪律](/zh/academy/benchmark-discipline) 与 [Benchmark 范围](/zh/validation/benchmark-scope) 确保你手上的数字本身值得解释。
- 再把 [Nsight Compute 中与 roofline 相关的资料](/zh/research/references#profiler、工具与诊断资料) 和 [性能模型类阅读](/zh/research/references#论文与性能心智模型) 对照起来看。
- 最后把观察结果重新映射回你正在阅读的 kernel 阶段：naive、tiled、double-buffered 还是 WMMA。

建议一直追问：

- 这次优化提升的是算术强度、降低的是延迟，还是只是把成本换了位置？
- 当前瓶颈更像是内存流量、指令混合，还是 launch 几何？
- 你凭什么认为下一步该追 Tensor Core，而不是继续优化内存路径？

## Tensor Core 约束与 fallback 设计

当 WMMA 在图表里看起来很亮眼，但放到真实 workload 里却显得脆弱时，走这条路线。

- 先读 [Tensor Core 路径](/zh/architecture/tensor-core-path) 与 [Tensor Core WMMA](/zh/academy/kernel-tensor-core)。
- 再打开 [WMMA API 与混合精度指南](/zh/research/references#cuda-与-nvidia-官方文档)，确认 fragment 大小、对齐要求、数据转换成本。
- 最后把这些约束和仓库中的 fallback 逻辑对照起来，理解为什么“不支持的 shape 应安全退回 FP32”是一种工程选择，而不是性能妥协。

建议一直追问：

- 哪些输入 shape 真正“适合 Tensor Core”，哪些不适合？
- 计时里有多少是真正的矩阵乘法，有多少是转换或 wrapper 成本？
- 什么时候 FP32 fallback 反而是更诚实的实现？

## 从症状到证据的 profiling 路线

当你只知道“结果变了”，却还不知道问题在时间线、单个 kernel，还是整体 benchmark 流程里时，走这条路线。

- 从 [诊断闭环](/zh/academy/diagnosis-loop) 开始。
- 配合 [Nsight Systems、Nsight Compute 与 Compute Sanitizer](/zh/research/references#profiler、工具与诊断资料)，先拆开 launch、内存与正确性问题。
- 再对照 [可复现性](/zh/validation/reproducibility)，确保 profiler 会话、benchmark 结果和环境描述是一致的。

建议一直追问：

- 症状体现在 timeline、单个 kernel 指标，还是最终 benchmark 汇总里？
- 哪个指标最能区分带宽瓶颈、occupancy 问题、延迟隐藏不足或错误假设？
- 如果要对外发布性能结论，你还缺什么证据？

## 按当前目标选路线

| 当前目标 | 最适合的路线 |
|---|---|
| 想真正理解 shared-memory tiling | [GEMM 分块与层级化思维](#gemm-分块与层级化思维) |
| 想更准确地讨论 occupancy | [把 occupancy 当约束，不当口号](#把-occupancy-当约束-不当口号) |
| 想用更强的模型解释性能上限 | [面向 SGEMM 的 Roofline 思维](#面向-sgemm-的-roofline-思维) |
| 想搞清楚 Tensor Core 什么时候值得用 | [Tensor Core 约束与 fallback 设计](#tensor-core-约束与-fallback-设计) |
| 想把 profiler 输出变成调试计划 | [从症状到证据的 profiling 路线](#从症状到证据的-profiling-路线) |
