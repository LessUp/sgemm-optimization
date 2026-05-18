---
title: 系统蓝图
---

# 系统蓝图

本页是 SGEMM 优化系统的完整组件级蓝图，映射了所有主要组件、它们之间的数据流，以及约束这些流的设计决策。

<ThemedFigure
  :wide="true"
  light="/figures/whitepaper-system-light.svg"
  dark="/figures/whitepaper-system-dark.svg"
  alt="展示 kernel 组件、数据路径以及附加到阶梯上的验证与研究辅助轨的完整系统蓝图。"
  caption="蓝图映射了每一个有明确存在理由的组件。没有理由的组件要么不在这里，要么被标注为待决问题。"
/>

## 组件清单

| 组件 | 作用 | 约束 |
|---|---|---|
| `main.cu` 入口 | 解析参数、选择 kernel 变体、路由到 benchmark 或正确性检查 | 不得硬编码变体，选择在运行时完成 |
| `kernels/sgemm_naive.cuh` | 基线 FP32，每个线程处理一个输出元素 | 建立代价模型；不使用共享内存 |
| `kernels/sgemm_tiled.cuh` | 使用共享内存 staging 的 tiled FP32 | tile 大小是编译期模板参数 |
| `kernels/sgemm_bank_free.cuh` | 通过 padding 消除 bank 冲突的 tiled FP32 | Padding 是与 tiled 版本的唯一结构差异 |
| `kernels/sgemm_double_buffer.cuh` | 使用双缓冲重叠 staging 与 compute | 共享内存中至少需要两个 staging buffer |
| `kernels/sgemm_tensor_core.cuh` | 基于 WMMA 的计算，带 FP32 入口路径 | 受设备能力和 shape 整除性保护 |
| `utils/cuda_check.h` | CUDA API 和 kernel 调用的 RAII 错误传播 | 失败时抛出 `std::runtime_error`，不静默返回 |
| `utils/matrix.h` | 主机侧矩阵初始化与参考计算 | 行主序布局；参考使用 CPU BLAS 或 naive FP64 循环 |
| `tests/test_sgemm.cu` | 基于 cuBLAS oracle 的正确性测试套件 | 只在 GPU 上运行；不包含在托管 CI 中 |
| Docs 站点 | 叙事层——架构、学院、验证、研究 | VitePress 双语路由；无运行时 GPU 依赖 |

## 数据流：从主机到设备

```
主机分配 A、B、C（行主序，FP32）
  │
  ▼
cudaMemcpy H→D
  │
  ▼
Kernel 启动（grid、block、共享内存预算）
  ├─ Naive 路径：每线程直接读取全局内存
  ├─ Tiled 路径：协作式 staging 到共享 tile
  ├─ Bank-free 路径：带 padding 的 tile staging
  ├─ Double-buffer 路径：异步预取下一个 tile
  └─ Tensor Core 路径：FP32→FP16 转换 + WMMA fragment 累加
  │
  ▼
cudaMemcpy D→H
  │
  ▼
与 cuBLAS oracle 正确性校验（仅本地 GPU）
```

## 设计决策及其架构影响

### RAII 错误处理

所有 CUDA API 调用和 kernel 启动都通过 `cudaCheck()` 包装。这确保任何失败路径都会立即以可追踪的错误终止，而不会把错误结果静默地传递下去。

**影响：** 测试代码不会意外吞掉错误然后用错误输出与 cuBLAS oracle 比较，避免了"失败的 kernel 看起来通过了测试"的问题。

### 运行时 kernel 选择

入口点从命令行参数在运行时选择 kernel 变体，而不是编译多个可执行文件。

**影响：** 变体之间的 benchmark 对比使用相同的二进制文件和相同的主机代码路径，对比更干净，消除了构建参数引入的干扰变量。

### 模板 tile 大小

tile 维度是编译期模板参数，而非运行时常量。

**影响：** 编译器在编译时就知道共享内存布局，能生成高效的寻址代码，避免动态共享内存分配开销。代价是未编译的 tile 大小无法在不重新构建的情况下进行 benchmark。

### Tensor Core 作为受保护的可选路径

Tensor Core 变体在提交 WMMA 计算之前检查设备能力和 shape 整除性，否则回退到 FP32 tiled 路径。

**影响：** 系统可以安全地在非 Tensor Core 硬件上运行，来自该硬件的 benchmark 结果被标记为 FP32 结果而非混合精度结果。

## 蓝图中的验证边界

蓝图明确区分了编译期可验证的约束和运行时可验证的约束：

| 约束类别 | 可验证环境 |
|---|---|
| 文件结构、文档、OpenSpec 对齐 | 托管 CI |
| CUDA 代码编译无警告 | 托管 CI（仅编译） |
| cuBLAS oracle 正确性 | 本地 GPU 运行 |
| Benchmark 数字和加速比 | 带命名硬件的本地 GPU 运行 |

## 相关页面

- [架构概述](./index) — 系统地图与设计原理
- [Kernel 阶梯](./kernel-ladder) — 优化顺序与瓶颈转移
- [Tensor Core 路径](./tensor-core-path) — Tensor Core 约束与回退行为
- [正确性策略](../validation/correctness-policy) — oracle 定义与容差阈值
- [性能模型](../validation/performance-model) — 量化代价模型
