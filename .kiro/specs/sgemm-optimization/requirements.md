# Requirements Document

## Introduction

本项目实现一个从零开始的 CUDA 矩阵乘法 (SGEMM) 优化演进系统。目标是展示从最简单的三层循环实现到接近 cuBLAS 性能的完整优化路径，涵盖 HPC 领域的核心优化技术：共享内存分块、Bank Conflict 消除、双缓冲流水线、以及 Tensor Core 加速。

## Glossary

- **SGEMM**: Single-precision General Matrix Multiply，单精度通用矩阵乘法 (C = α*A*B + β*C)
- **Kernel**: CUDA 中运行在 GPU 上的并行函数
- **Global_Memory**: GPU 的全局显存，带宽高但延迟大
- **Shared_Memory**: GPU SM 内的片上共享内存，延迟低但容量有限
- **Coalescing**: 全局显存合并访问，多个线程访问连续地址时合并为一次事务
- **Bank_Conflict**: 共享内存 bank 冲突，多个线程访问同一 bank 导致串行化
- **Tiling**: 分块技术，将大矩阵分成小块在共享内存中处理
- **Double_Buffering**: 双缓冲技术，使用两个缓冲区实现计算与访存重叠
- **Tensor_Core**: NVIDIA GPU 的专用矩阵计算单元，支持混合精度矩阵运算
- **WMMA**: Warp Matrix Multiply Accumulate，Tensor Core 的编程接口
- **Roofline_Model**: 性能分析模型，展示计算密度与性能上限的关系
- **GFLOPS**: 每秒十亿次浮点运算，性能度量单位
- **Benchmark_System**: 性能测试系统，用于测量和比较不同实现的性能

## Requirements

### Requirement 1: Naive SGEMM 实现

**User Story:** As a developer, I want to implement a basic three-loop SGEMM kernel, so that I can establish a performance baseline and understand the fundamental algorithm.

#### Acceptance Criteria

1. THE Naive_Kernel SHALL compute C = A * B where A is M×K, B is K×N, and C is M×N matrices
2. WHEN the Naive_Kernel executes, THE system SHALL assign one thread per output element C[i][j]
3. THE Naive_Kernel SHALL produce numerically correct results within floating-point tolerance (relative error < 1e-5)
4. THE Benchmark_System SHALL measure and report GFLOPS for the Naive_Kernel
5. THE system SHALL support matrix dimensions that are multiples of 32

### Requirement 2: Shared Memory Tiling 优化

**User Story:** As a developer, I want to optimize global memory access using shared memory tiling, so that I can reduce global memory bandwidth consumption and improve performance.

#### Acceptance Criteria

1. THE Tiled_Kernel SHALL load matrix tiles into Shared_Memory before computation
2. WHEN loading tiles, THE Tiled_Kernel SHALL use coalesced global memory access patterns
3. THE Tiled_Kernel SHALL synchronize threads after loading each tile using __syncthreads()
4. THE Tiled_Kernel SHALL produce numerically correct results matching the Naive_Kernel output
5. THE Benchmark_System SHALL demonstrate performance improvement over Naive_Kernel
6. THE Tiled_Kernel SHALL use configurable tile sizes (default 32×32)

### Requirement 3: Bank Conflict 消除

**User Story:** As a developer, I want to eliminate shared memory bank conflicts, so that I can maximize shared memory bandwidth utilization.

#### Acceptance Criteria

1. THE BankConflict_Free_Kernel SHALL use padding or transposed storage to avoid bank conflicts
2. WHEN accessing Shared_Memory, THE BankConflict_Free_Kernel SHALL ensure different threads in a warp access different banks
3. THE BankConflict_Free_Kernel SHALL produce numerically correct results
4. THE Benchmark_System SHALL demonstrate performance improvement over Tiled_Kernel
5. THE system SHALL provide documentation explaining the bank conflict resolution strategy

### Requirement 4: Double Buffering 流水线

**User Story:** As a developer, I want to implement double buffering to overlap computation with memory access, so that I can hide memory latency and improve throughput.

#### Acceptance Criteria

1. THE DoubleBuffer_Kernel SHALL use two sets of shared memory buffers
2. WHEN computing on one buffer, THE DoubleBuffer_Kernel SHALL prefetch the next tile into the other buffer
3. THE DoubleBuffer_Kernel SHALL properly synchronize buffer switching to avoid data hazards
4. THE DoubleBuffer_Kernel SHALL produce numerically correct results
5. THE Benchmark_System SHALL demonstrate performance improvement over BankConflict_Free_Kernel

### Requirement 5: Tensor Core (WMMA) 加速

**User Story:** As a developer, I want to leverage Tensor Cores using WMMA API, so that I can achieve maximum performance approaching cuBLAS levels.

#### Acceptance Criteria

1. THE TensorCore_Kernel SHALL use NVIDIA WMMA API for matrix multiply-accumulate operations
2. THE TensorCore_Kernel SHALL handle matrix dimensions compatible with WMMA fragment sizes (16×16×16)
3. THE TensorCore_Kernel SHALL produce numerically correct results within Tensor Core precision tolerance
4. THE Benchmark_System SHALL compare TensorCore_Kernel performance against cuBLAS
5. THE TensorCore_Kernel SHALL achieve at least 70% of cuBLAS performance on supported hardware

### Requirement 6: 性能基准测试系统

**User Story:** As a developer, I want a comprehensive benchmarking system, so that I can measure, compare, and visualize the performance evolution across all implementations.

#### Acceptance Criteria

1. THE Benchmark_System SHALL measure execution time using CUDA events with proper warm-up runs
2. THE Benchmark_System SHALL calculate and report GFLOPS for each kernel implementation
3. THE Benchmark_System SHALL verify numerical correctness against a reference implementation
4. THE Benchmark_System SHALL test multiple matrix sizes (512, 1024, 2048, 4096)
5. THE Benchmark_System SHALL output results in a format suitable for Roofline_Model analysis
6. THE Benchmark_System SHALL compare all implementations against cuBLAS as the performance ceiling

### Requirement 7: 正确性验证

**User Story:** As a developer, I want robust correctness verification, so that I can ensure all optimized kernels produce accurate results.

#### Acceptance Criteria

1. THE Verification_System SHALL compare kernel output against cuBLAS reference results
2. THE Verification_System SHALL report maximum absolute error and relative error
3. WHEN relative error exceeds 1e-4 for standard kernels, THE Verification_System SHALL flag the result as incorrect
4. WHEN relative error exceeds 1e-3 for Tensor Core kernels, THE Verification_System SHALL flag the result as incorrect (due to mixed precision)
5. THE Verification_System SHALL test edge cases including non-square matrices
