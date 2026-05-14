---
title: Benchmark 纪律
---

# Benchmark 纪律

Benchmark 纪律的目标，是让每一次实验都能教会你一件事。

仓库已经提供了三种常用模式：固定单一 shape 做隔离、使用默认混合 shape 集合找回归、以及拉长测量窗口做公开汇报。不要把它们当成同一种证据。

## 先写假设，再写命令

运行前先写清三件事：

1. **你要测哪个 shape**
2. **你认为自己在测哪个瓶颈**
3. **你准备改哪一个点来移动这个瓶颈**

如果这三项说不清，这次 run 仍然属于探索，不应该被写成“结果”。

## 规范实验模板

### 固定单一 shape

```bash
./build/bin/sgemm_benchmark --dims 1024 1024 1024
```

适合去掉 shape 多样性，只在稳定上下文里观察一个瓶颈。

### 扫描仓库默认测试集

```bash
./build/bin/sgemm_benchmark -a
```

默认集合刻意混合了：

- 对齐的方阵：`512`、`1024`
- 一个对齐但非方阵案例：`256 x 384 x 640`
- 一个不规则边界案例：`511 x 513 x 1025`

适合判断改动是否稳健，还是只在“友好维度”上好看。

### 提升测量可信度

```bash
./build/bin/sgemm_benchmark -a --warmup 10 --benchmark 50
```

默认值是 `--warmup 5 --benchmark 20`。在写文档、PR 总结或公开对比前，应该把测量窗口拉长。

## Shape 选择规则

| 场景 | 优先选择 | 原因 |
|------|----------|------|
| 你在追一个单点回归 | 一个明确的 `--dims` | 噪声更小，迭代更快 |
| 你在检查 launch 改动能否泛化 | `-a` | 能同时暴露对齐与不规则行为 |
| 你在讨论 WMMA | 至少一个 16 对齐 shape + 一个不规则 shape | 同时观察快路径潜力与 fallback 代价 |
| 你在给最终数字 | 标准 shape + 不规则 shape | 防止把“友好案例”误写成通用承诺 |

## 保持实验诚实的规则

- **一次循环只验证一个假设。** 同时改 tile、stage 深度和 fallback 策略，benchmark 学不到结论。
- **不要跳过正确性复核。** `ctest --test-dir build` 是闭环的一部分，不是收尾动作。
- **立刻标注 benchmark 范围。** 先判断这次看到的是端到端还是 compute-only，再交给 [Benchmark 范围](/zh/validation/benchmark-scope) 解释。
- **结果必须带环境信息。** GPU 型号、CUDA 版本、维度、warmup 次数与 benchmark 次数，应和结果一起记录。

## 公开汇报前检查表

- 运行 `ctest --test-dir build`
- 同时比较一个标准 shape 和一个不规则 shape
- 确认结果是端到端还是 compute-only
- 记录完整 benchmark 命令
- 在把结果写成仓库结论前，先交给 [验证](/zh/validation/)
