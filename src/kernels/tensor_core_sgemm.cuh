#pragma once

// ============================================================================
// Tensor Core SGEMM - 公共接口
// ============================================================================
//
// 此文件作为 Tensor Core 功能的公共入口点。
// 内部实现拆分为多个深度模块：
//
// 1. tensor_core_capabilities.cuh - 能力查询接口
//    - tensorCoresAvailable()
//    - tensorCoreDimensionsSupported()
//    - getTensorCoreArchName()
//
// 2. tensor_core_compute.cuh - 纯 WMMA 计算路径
//    - float_to_half_kernel
//    - launch_tensor_core_sgemm_fp16()
//    - launch_tensor_core_sgemm_fp16_fast_path()
//
// 3. tensor_core_launcher.cuh - 统一启动接口（强制显式 fallback）
//    - launch_tensor_core_sgemm_with_fallback()
//    - FallbackKernel 类型定义
//    - Tensor Core 容差常量
//
// 4. tensor_core_benchmark.cuh - Tensor Core 专用 benchmark
//    - runTensorCoreComputeOnlyBenchmark()
//
// 设计原则：
// - 不提供默认 fallback，强制调用者显式指定
// - 消除与具体内核的循环依赖
// - 每个模块有独立的测试面
// ============================================================================

#include "tensor_core_capabilities.cuh"
#include "tensor_core_compute.cuh"
#include "tensor_core_launcher.cuh"
#include "tensor_core_benchmark.cuh"
