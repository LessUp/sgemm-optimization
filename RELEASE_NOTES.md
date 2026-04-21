# SGEMM Optimization v2.1.0 Release Notes

## English

### Changed

- **Documentation Refactor**: Complete documentation restructure
  - Consolidated changelog entries into single `CHANGELOG.md`
  - Rewrote `.kiro/specs/` documentation for better clarity
  - Enhanced `index.md` GitHub Pages landing page with:
    - Performance comparison tables
    - ASCII optimization roadmap
    - Quick start guide
    - GPU architecture reference
  - Updated `_config.yml` for optimized Jekyll configuration
  - **GitHub Workflows**: Simplified and optimized
    - `ci.yml`: Cleaner structure, better step naming
    - `pages.yml`: Fixed paths filter, improved concurrency

### Removed

- **Dead Code Cleanup**: Removed 514 lines of unused code across 7 source files
  - `src/utils/cuda_utils.cuh`: Removed unused utility functions
  - `src/utils/verify.cuh`: Removed unused verification functions
  - All kernel files: Removed unused alternative implementations

## 中文

### 变更

- **文档重构**: 完整文档重构
  - 整合所有变更记录到单一 `CHANGELOG.md`
  - 重写 `.kiro/specs/` 文档，提升可读性
  - 增强 `index.md` GitHub Pages 首页：
    - 📊 性能对比表格
    - 🔄 优化演进路线图（ASCII）
    - 🚀 快速开始指南
    - 📖 GPU 架构支持表
  - 优化 `_config.yml` Jekyll 配置
  - **GitHub Workflows**: 简化并优化工作流
    - `ci.yml`: 清晰的步骤命名
    - `pages.yml`: 修复 paths 过滤，改进并发配置

### 移除

- **死代码清理**: 删除 514 行未使用的代码
  - `src/utils/cuda_utils.cuh`: 移除未使用的工具函数
  - `src/utils/verify.cuh`: 移除未使用的验证函数
  - 所有 kernel 文件: 移除未使用的替代实现

## Upgrade Guide

```bash
# 克隆最新版本
git clone https://github.com/LessUp/sgemm-optimization.git

# CMake 构建
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 运行基准测试
./build/bin/sgemm_benchmark -a
```

## Highlights

- 📚 **Improved Documentation**: Complete restructure of all documentation
- 🧹 **Cleaner Codebase**: Removed 514 lines of unused code
- 🚀 **Better Developer Experience**: Enhanced GitHub Pages with comprehensive guides
- 🔧 **Optimized CI**: Simplified workflows for better maintainability
