# SGEMM Optimization 领域模型

本文档定义了项目的核心概念和术语，供 AI 工具和人类开发者参考。

## 核心模块

### Kernel Catalog 模块

**位置**: `src/kernels/kernel_catalog.cuh`

**权威元数据源** - 内核阶梯的唯一事实来源：

- **KernelCatalogEntry**: 完整的内核元数据
  - `name`: 显示名称
  - `type`: KernelType::Standard 或 KernelType::TensorCore
  - `launcher`: 启动适配器
  - `constraints`: 运行时约束（Tensor Core 要求、维度对齐）
- **KernelConstraints**: 运行时约束描述
  - `requires_tensor_cores`: 是否需要 sm_70+
  - `dimension_alignment`: 维度对齐要求（0 = 无约束）
  - `requires_compute_only`: 是否使用特殊 benchmark 接口
- **查询工具**: `countKernelsByType()`, `getKernelNames()`, `canRunTensorCoreKernels()`

**设计原则**：
- **单一事实源**: 新增内核只需添加一个 catalog 条目
- **自描述约束**: 每个 entry 知道自己能否在给定条件下运行
- **统一调度**: BenchmarkRunner 通过 catalog 迭代，无特殊分支

### Tensor Core 模块

**位置**: `src/kernels/tensor_core_sgemm.cuh`

深层模块，提供完整的 Tensor Core SGEMM 功能：

- **能力查询**: `tensorCoresAvailable()`, `tensorCoreDimensionsSupported()`, `getTensorCoreArchName()`
- **计算内核**: `float_to_half_kernel`, `launch_tensor_core_sgemm_fp16()`, `launch_tensor_core_sgemm_fp16_fast_path()`
- **统一接口**: `launch_tensor_core_sgemm_with_fallback()` - 端到端 FP32 入口点（强制显式指定 fallback）
- **容差常量**: `kTensorCoreVerifyTolerance` - Tensor Core 验证容差（FP16 中间精度）

**设计原则**：
- **深度提升**：小接口（能力查询 + 启动入口），大实现（WMMA 内核 + 类型转换 + fallback 逻辑）
- **不提供默认 fallback**：调用者必须显式指定 fallback 策略
- **编译期解耦**：Tensor Core 模块不依赖任何具体内核

#### Tensor Core Benchmark
**位置**: `src/kernels/tensor_core_benchmark.cuh`

Tensor Core 特有的 benchmark 功能，提供：
- `canRunTensorCoreComputeOnly()` - 约束检查（与 KernelCatalog 语义一致）
- `runTensorCoreComputeOnlyBenchmark()` - 纯计算路径性能测试

**接口设计**：只接受 `cublasHandle_t`，不依赖整个 `SGEMMBenchmark` 类，避免内核层对工具层的上穿依赖。

## 验证模块

**位置**: `src/utils/verify.cuh`

**统一的验证策略** - reference + comparison + tolerance policy：

- **VerifyResult**: 验证结果结构（pass/fail、错误指标）
- **VerifyTolerance**: 容差规范（numpy-style allclose 语义）
  - `kStandardVerifyTolerance`: FP32 标准容差
  - `kTensorCoreVerifyTolerance`: Tensor Core 宽松容差
- **比较函数**:
  - `compareMatrices()`: Host 指针比较
  - `compareDeviceMatrices()`: Device 指针比较
- **SGEMMVerifier**: cuBLAS 参考计算适配器
  - `computeReference()`: 计算参考结果
  - `verify()`, `verifyDevice()`: 验证内核输出

**设计原则**：
- **单一验证政策**: 所有内核共享同一套容差语义
- **分离关注点**: 参考计算 vs 比较逻辑
- **可扩展**: 未来可添加其他参考适配器

## Benchmark 模块

项目将 Benchmark 功能拆分为三个深度模块，每个模块有独立的职责：

### Benchmark Settings
**位置**: `src/utils/benchmark_settings.cuh**

配置集中化：
- `RunSettings`: 预热次数、测量次数
- `VerificationSettings`: 容差配置
- `OutputSettings`: Roofline 导出选项
- `BenchmarkSettings`: 聚合配置

### Benchmark Core
**位置**: `src/utils/benchmark_core.cuh`

核心性能测量：
- `BenchmarkResult`: 结果结构和报告生成
- `CudaTimer` - RAII 包装的 CUDA 事件计时器
- `measureGpuTime()` - 通用的 GPU 操作性能测量器

### Benchmark Metrics
**位置**: `src/utils/benchmark_metrics.cuh`

指标计算：
- `PerformanceMetrics` - 性能指标结构体
- `calculateSgemmMetrics()` - 计算 SGEMM 性能指标
- `getTheoreticalPeakGflops()` / `getTheoreticalPeakBandwidth()` - 理论峰值查询
- `calculateEfficiency()` / `calculateBandwidthUtilization()` - 效率计算

### 高级接口
**位置**: `src/utils/benchmark.cuh`

聚合模块并提供：
- `SGEMMBenchmark` - 高级 benchmark 编排器

## 测试架构

### 测试分层

项目采用分层测试策略，确保每个层级都有独立的测试面：

#### CPU-only 测试
**位置**: `tests/test_benchmark_settings.cpp`, `tests/test_device_info_cpu.cpp`

纯 CPU 测试，不需要 CUDA 设备：
- 设置模块单元测试
- 设备信息 Seam 测试（使用 fake provider）

#### 内核层测试
**位置**: `tests/test_sgemm.cu`

测试内核的正确性：
- 参数化正确性测试（5 个内核 + 多维度组合）
- Tensor Core 快速路径和 fallback 测试
- 边界测试和维度不变性测试

#### Kernel Catalog 测试
**位置**: `tests/test_kernel_catalog.cu`

测试内核目录的元数据和约束：
- Catalog 包含预期的内核
- 条目有有效的元数据（名称、启动器、约束）
- 约束检查正确工作

#### 工具层测试
**位置**: `tests/test_utils.cu`

测试工具模块的独立接口：
- `DeviceMemory` - RAII 内存管理、移动语义、边界条件
- `CublasHandle` - cuBLAS 句柄生命周期
- `SGEMMVerifier` - 参考计算和验证逻辑
- `VerifyTolerance` - 容差配置和边界条件
- NaN/Inf 处理、异常安全性

#### 性能回归测试
**位置**: `tests/test_performance.cu`

检测性能退化：
- 为每个内核定义最小性能阈值（相对于理论峰值的百分比）
- 测量实际 GFLOPS 并与阈值比较
- 使用固定阈值策略，无需外部基线文件

**性能阈值**：
- Naive: 5% 峰值
- Tiled: 20% 峰值
- Bank-Conflict-Free: 30% 峰值
- Double-Buffer: 35% 峰值
- Tensor Core: 50% 峰值（当可用时）

### 测试分类标签

项目使用 CTest labels 区分测试类型：
- `cpu`: CPU-only 测试，不需要 CUDA 设备
- `cuda`: 需要 CUDA 设备的测试，无 GPU 时跳过
- `performance`: 性能回归测试

**运行命令**:
```bash
ctest -L cpu          # 只运行 CPU 测试
ctest -L cuda         # 只运行 CUDA 测试
ctest -L performance  # 只运行性能测试
```

## 架构原则

### 三层架构

1. **应用层** (`main.cu`, `cli_parser.cuh`, `benchmark_runner.cuh`)
   - `main.cu` - 入口点，仅负责组装
   - `cli_parser.cuh` - 命令行解析、配置构造
   - `benchmark_runner.cuh` - 内核调度、结果聚合
2. **内核层** (`src/kernels/`) - 5 个内核实现 + Kernel Catalog + Tensor Core 专用模块
3. **工具层** (`src/utils/`) - RAII 内存管理、错误处理、验证辅助、设置模块

### 依赖方向

- 应用层 → 内核层 → 工具层
- 内核层可以依赖工具层
- 工具层不应依赖内核层（通过适配器解耦）

### 模块深度原则

- **深层模块**: 小接口，大实现（高杠杆）
- **浅层模块**: 接口复杂度接近实现复杂度（应避免或合并）

## 性能测试维度

| 内核 | 文件 | 优化技术 |
|------|------|----------|
| Naive | `naive_sgemm.cuh` | 基础三重循环，基准实现 |
| Tiled | `tiled_sgemm.cuh` | 共享内存分块，数据复用 |
| Bank-Free | `bank_conflict_free_sgemm.cuh` | 共享内存填充，消除 bank 冲突 |
| Double-Buffer | `double_buffer_sgemm.cuh` | 双缓冲，计算与传输重叠 |
| Tensor Core | `tensor_core_*.cuh` | WMMA API，混合精度 FP16→FP32 |
