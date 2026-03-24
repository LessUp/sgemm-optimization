# Contributing

感谢你对本项目的关注！欢迎通过 Issue 和 Pull Request 参与贡献。

## 开发流程

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m "feat: add your feature"`
4. 推送分支：`git push origin feature/your-feature`
5. 创建 Pull Request

## 构建与测试

推荐优先使用 CMake：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/bin/sgemm_benchmark
cmake --build build --target test_sgemm
ctest --test-dir build
```

Makefile 也可用于快速本地构建：

```bash
make GPU_ARCH=sm_86
make benchmark
make test
```

说明：GitHub Actions 当前执行格式检查和容器化 CUDA compile-only 构建；CUDA 运行时测试仍需在本地或带 GPU 的 runner 上完成。

## 代码规范

- CUDA 代码遵循项目现有风格
- 使用 `.editorconfig` 中定义的缩进和格式规则
- 新增 kernel 版本请附带正确性验证（vs cuBLAS）
- 确保所有现有测试通过

## 提交信息格式

推荐使用 [Conventional Commits](https://www.conventionalcommits.org/)：

- `feat:` 新功能 / 新 kernel 版本
- `fix:` 修复 Bug
- `perf:` 性能优化
- `docs:` 文档更新
- `test:` 测试相关
