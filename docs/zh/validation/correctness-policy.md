---
title: 正确性策略
---

# 正确性策略

任何性能工作，只有在正确性仍然满足仓库 oracle 的前提下才会被接受。

## Oracle 与基线

项目使用 **cuBLAS SGEMM** 作为运行时验证的参考实现。所有 kernel 输出都与这个参考结果对照，而不是与另一个项目内 kernel 互相比对。

原因很直接：优化工作经常会改变 launch 几何、staging 方式和混合精度路径。仅靠项目内部基线不足以支撑这些检查。

## 容差策略

验证器采用 NumPy 风格的 `allclose` 规则：

```text
|test - ref| <= atol + rtol * |ref|
```

### 标准 FP32 kernel

- `rtol = 1e-3`
- `atol = 1e-4`

这些阈值适用于 Naive、Tiled、Bank Conflict Free 与 Double Buffer 路径。

### Tensor Core / 混合精度路径

- `rtol = 5e-2`
- `atol = 1e-2`

Tensor Core 路径使用更宽松的容差，因为它包含混合精度行为。

## Shape 覆盖要求

测试套件并不只覆盖“友好方阵”。

- 标准维度集合同时包含小尺寸、方阵、矩形与不规则 shape
- Tensor Core 快路径集合包含 16 对齐案例
- Tensor Core fallback 集合刻意包含 `15x15x15`、`17x19x23`、`511x513x1025` 这类不规则维度

如果一次性能改动只在友好子集上成立，却悄悄削弱了 fallback 路径，这个结果就不值得信任。

## CI 与本地的职责边界

托管 CI 可以证明结构、文档流程和治理一致性，但**不能**证明 CUDA kernel 的运行时正确性。

本地 GPU 执行必须运行：

```bash
ctest --test-dir build
```

只有这一步通过后，基于 benchmark 的性能陈述才值得进入评审。
