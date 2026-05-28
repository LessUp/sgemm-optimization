---
title: 快速上手
---

# 快速上手

编译、运行和验证项目，无需猜测工具链



## 推荐编译流程

```bash
git clone https://github.com/LessUp/sgemm-optimization.git
cd sgemm-optimization

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

运行默认 benchmark：

```bash
./build/bin/sgemm_benchmark
```

运行完整 benchmark 集：

```bash
./build/bin/sgemm_benchmark -a
```

运行测试：

```bash
ctest --test-dir build
```



## 验证边界

| 环境 | 运行什么 |
|------|----------|
| 本地 GPU 机器 | benchmark、运行时验证、`ctest` |
| 托管 CI | 格式检查、CUDA 编译、文档测试/构建、路由检查、Pages |

这种划分是刻意的：GitHub 托管 runner 验证仓库健康，而性能和 CUDA 运行时正确性仍需真正的 GPU 机器。



## 接下来去哪

- [学习路径](/zh/academy/learning-path)
- [架构概览](/zh/architecture/)
- [验证概览](/zh/validation/)
