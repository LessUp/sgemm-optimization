---
title: 可复现性
---

# 可复现性

本仓库中的“可复现”，意味着另一位读者能够判断**运行了什么、在哪台机器上运行、以及这个结果能支撑哪一种结论**。

## 最小本地流程

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build
./build/bin/sgemm_benchmark -a --warmup 10 --benchmark 50
```

这是形成一条本地性能陈述的最低门槛。

## 必须记录的环境信息

每次公开结果都应该带上：

- GPU 型号
- CUDA toolkit / driver 环境
- benchmark 命令
- 使用的维度或 benchmark 集合
- warmup 次数与 benchmark 次数
- 该数字是端到端还是 compute-only

缺少这些元数据时，读者无法判断这个数字是否可与公开快照直接对比。

## 托管 CI 与本地复跑

托管 CI 依然重要，因为它证明文档、Pages 与治理表面仍然连贯。但 CI runner 不是运行时行为的证据来源。

只有本地 GPU 复跑才能确认：

- 针对当前机器，结果仍然通过 cuBLAS 正确性对照
- Tensor Core 快路径条件是否真的满足
- 测得的收益是否经得起当前工作负载组合

## 汇报检查表

在公开或复述一个结果之前，先确认你能回答：

- 这个数字来自哪块 GPU？
- 这个数字对应哪条命令？
- 这个数字属于哪个 benchmark 标签？
- 这个 benchmark 之前通过了哪次正确性检查？
- 哪个不规则 shape 防止这条结论变成“只挑友好案例”的 cherry-pick？

如果回答不出来，请先重跑实验，再引用数字。
