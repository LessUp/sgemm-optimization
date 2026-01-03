# Implementation Plan: SGEMM Optimization

## Overview

本实现计划将 SGEMM 优化设计转化为可执行的编码任务。采用渐进式实现策略，从基础设施开始，逐步实现各优化版本，每个阶段都包含验证和测试。

## Tasks

- [x] 1. 项目基础设施搭建
  - [x] 1.1 创建项目目录结构和 Makefile
    - 创建 src/kernels/, src/utils/, tests/ 目录
    - 编写支持 CUDA 编译的 Makefile（包含 cuBLAS 链接）
    - _Requirements: 6.1, 6.6_
  - [x] 1.2 实现 CUDA 工具函数 (cuda_utils.cuh)
    - 实现 CUDA_CHECK 和 CUBLAS_CHECK 宏
    - 实现设备内存分配/释放的 RAII 包装器
    - 实现矩阵初始化和随机数生成函数
    - _Requirements: 1.5_

- [x] 2. 验证系统实现
  - [x] 2.1 实现 cuBLAS 参考计算 (verify.cuh)
    - 封装 cublasSgemm 调用作为参考实现
    - 实现结果比较函数（计算最大绝对/相对误差）
    - 实现正确性判断函数（根据阈值判断）
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  - [x] 2.2 编写验证系统属性测试
    - **Property 3: Error Detection Correctness**
    - **Validates: Requirements 7.3, 7.4**

- [x] 3. Naive SGEMM 实现
  - [x] 3.1 实现 Naive Kernel (naive_sgemm.cuh)
    - 实现基础三层循环 kernel
    - 每个线程计算一个输出元素
    - 实现 kernel 启动包装函数
    - _Requirements: 1.1, 1.2, 1.3_
  - [x] 3.2 编写 Naive Kernel 正确性属性测试
    - **Property 1: Kernel Numerical Correctness (Naive)**
    - **Validates: Requirements 1.1, 1.3**

- [x] 4. Checkpoint - 基础验证
  - 确保 Naive Kernel 通过所有测试
  - 验证与 cuBLAS 结果一致
  - 如有问题请询问用户

- [x] 5. Tiled SGEMM 实现
  - [x] 5.1 实现 Tiled Kernel (tiled_sgemm.cuh)
    - 实现共享内存分块加载
    - 实现 tile 遍历和部分和累加
    - 添加 __syncthreads() 同步
    - 支持可配置的 TILE_SIZE（默认 32）
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.6_
  - [x] 5.2 编写 Tiled Kernel 正确性属性测试
    - **Property 1: Kernel Numerical Correctness (Tiled)**
    - **Validates: Requirements 2.4**

- [x] 6. Bank Conflict Free SGEMM 实现
  - [x] 6.1 实现 Bank Conflict Free Kernel (bank_conflict_free_sgemm.cuh)
    - 在共享内存声明中添加 padding (+1)
    - 保持其他逻辑与 Tiled 版本一致
    - 添加注释说明 bank conflict 消除原理
    - _Requirements: 3.1, 3.2, 3.3, 3.5_
  - [x] 6.2 编写 Bank Conflict Free Kernel 正确性属性测试
    - **Property 1: Kernel Numerical Correctness (BankConflictFree)**
    - **Validates: Requirements 3.3**

- [x] 7. Double Buffer SGEMM 实现
  - [x] 7.1 实现 Double Buffer Kernel (double_buffer_sgemm.cuh)
    - 声明双缓冲区 As[2] 和 Bs[2]
    - 实现预加载第一个 tile
    - 实现计算与预取重叠的主循环
    - 正确处理缓冲区切换和同步
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - [x] 7.2 编写 Double Buffer Kernel 正确性属性测试
    - **Property 1: Kernel Numerical Correctness (DoubleBuffer)**
    - **Validates: Requirements 4.4**

- [x] 8. Checkpoint - 中期验证
  - 确保所有标准 Kernel 通过正确性测试
  - 验证性能逐步提升
  - 如有问题请询问用户

- [x] 9. Tensor Core SGEMM 实现
  - [x] 9.1 实现 Tensor Core Kernel (tensor_core_sgemm.cuh)
    - 包含 mma.h 头文件
    - 实现 FP32 到 FP16 的数据转换
    - 使用 WMMA API 实现矩阵乘法
    - 处理 16x16x16 fragment 对齐
    - _Requirements: 5.1, 5.2, 5.3_
  - [x] 9.2 编写 Tensor Core Kernel 正确性属性测试
    - **Property 2: Tensor Core Kernel Correctness**
    - **Validates: Requirements 5.3**

- [x] 10. 基准测试系统实现
  - [x] 10.1 实现 Benchmark 系统 (benchmark.cuh)
    - 实现 CUDA Event 计时（包含 warm-up）
    - 实现 GFLOPS 计算
    - 实现结果输出格式化
    - 实现 Roofline 数据导出
    - _Requirements: 6.1, 6.2, 6.4, 6.5_
  - [x] 10.2 实现主程序 (main.cu)
    - 集成所有 kernel
    - 运行完整基准测试
    - 输出性能对比表格
    - 与 cuBLAS 对比
    - _Requirements: 6.3, 6.6_

- [x] 11. 维度不变性测试
  - [x] 11.1 编写维度不变性属性测试
    - **Property 4: Dimension Invariance**
    - **Validates: Requirements 1.5, 2.6**

- [x] 12. Final Checkpoint - 完整验证
  - 确保所有测试通过
  - 验证性能演进符合预期
  - 生成最终性能报告
  - 如有问题请询问用户

## Notes

- 所有测试任务均为必须完成，确保每个 Kernel 都有完整的属性测试覆盖
- 每个 Kernel 实现后立即进行正确性验证
- Checkpoint 任务用于阶段性验证和用户确认
- 属性测试使用随机矩阵，每个属性至少运行 100 次迭代
- Tensor Core 需要 Volta 或更新架构的 GPU (sm_70+)
