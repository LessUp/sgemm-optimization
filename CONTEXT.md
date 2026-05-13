# SGEMM Optimization 领域模型

本文档定义了项目的核心概念和术语，供 AI 工具和人类开发者参考。

## 核心模块

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
- `runTensorCoreComputeOnlyBenchmark()` - 纯计算路径性能测试

**接口设计**：只接受 `cublasHandle_t`，不依赖整个 `SGEMMBenchmark` 类，避免内核层对工具层的上穿依赖。

## 验证模块

**位置**: `src/utils/verify.cuh`

统一的验证逻辑：
- `detail::compareMatricesImpl()` - 内部实现，供其他函数共享
- `compareMatrices()` - 独立的矩阵比较函数
- `SGEMMVerifier` - 带 cuBLAS 句柄的验证器类

## Benchmark 模块

项目将 Benchmark 功能拆分为三个深度模块，每个模块有独立的职责：

### Benchmark Core
**位置**: `src/utils/benchmark_core.cuh`

核心性能测量：
- `CudaTimer` - RAII 包装的 CUDA 事件计时器
- `measureGpuTime()` - 通用的 GPU 操作性能测量器

### Benchmark Metrics
**位置**: `src/utils/benchmark_metrics.cuh`

指标计算：
- `PerformanceMetrics` - 性能指标结构体
- `calculateSgemmMetrics()` - 计算 SGEMM 性能指标
- `getTheoreticalPeakGflops()` / `getTheoreticalPeakBandwidth()` - 理论峰值查询
- `calculateEfficiency()` / `calculateBandwidthUtilization()` - 效率计算

### Benchmark cuBLAS
**位置**: `src/utils/benchmark_cublas.cuh`

cuBLAS 参考实现：
- `CublasSgemm` - cuBLAS SGEMM 参考调用器
- `SgemmReferenceCalculator` - 完整参考计算流程

### 高级接口
**位置**: `src/utils/benchmark.cuh`

聚合模块并提供：
- `SGEMMBenchmark` - 高级 benchmark 编排器
- `BenchmarkResult` - 结果结构和报告生成

## 测试架构

### 测试分层

项目采用分层测试策略，确保每个层级都有独立的测试面：

#### 内核层测试
**位置**: `tests/test_sgemm.cu`

测试内核的正确性：
- 参数化正确性测试（5 个内核 + 多维度组合）
- Tensor Core 快速路径和 fallback 测试
- 边界测试和维度不变性测试

#### 工具层测试
**位置**: `tests/test_utils.cu`

测试工具模块的独立接口：
- `DeviceMemory` - RAII 内存管理、移动语义、边界条件
- `CublasHandle` - cuBLAS 句柄生命周期
- `SGEMMVerifier` - 参考计算和验证逻辑
- `VerifyTolerance` - 容差配置和边界条件
- NaN/Inf 处理、异常安全性

**设计原则**：工具层测试独立于内核测试，可以单独捕获工具类 bug。

#### 性能回归测试
**位置**: `tests/test_performance.cu`

检测性能退化：
- 为每个内核定义最小性能阈值（相对于理论峰值的百分比）
- 测量实际 GFLOPS 并与阈值比较
- 支持基线数据持久化（存储在 `tests/baselines/`）

**性能阈值**：
- Naive: 5% 峰值
- Tiled: 20% 峰值
- Bank-Conflict-Free: 30% 峰值
- Double-Buffer: 35% 峰值
- Tensor Core: 50% 峰值（当可用时）

**设计原则**：性能测试独立于正确性测试，可在 CI 中检测重大性能退化。

## 架构原则

### 三层架构

1. **应用层** (`main.cu`, `cli_parser.cuh`, `benchmark_runner.cuh`)
   - `main.cu` - 入口点，仅负责组装
   - `cli_parser.cuh` - 命令行解析、配置构造
   - `benchmark_runner.cuh` - 内核调度、结果聚合
2. **内核层** (`src/kernels/`) - 5 个内核实现 + Tensor Core 专用模块
3. **工具层** (`src/utils/`) - RAII 内存管理、错误处理、验证辅助

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
