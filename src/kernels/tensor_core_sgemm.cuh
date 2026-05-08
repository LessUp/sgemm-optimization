#pragma once

// ============================================================================
// Tensor Core SGEMM - 公共接口
// ============================================================================
//
// 此文件作为 Tensor Core 功能的公共入口点，保持向后兼容。
// 内部实现已拆分为多个深度模块：
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
// 3. tensor_core_launcher.cuh - 统一启动接口（支持自定义 fallback）
//    - launch_tensor_core_sgemm_with_fallback()
//    - FallbackKernel 类型定义
//    - Tensor Core 容差常量
//
// 4. tensor_core_launcher_impl.cuh - 默认实现
//    - launch_tensor_core_sgemm() 使用 bank-conflict-free 作为默认 fallback
//
// 5. tensor_core_benchmark.cuh - Tensor Core 专用 benchmark
//    - runTensorCoreComputeOnlyBenchmark()
//
// 这种拆分提高了：
// - 局部性 (Locality)：每个关注点有独立的测试面
// - 杠杆 (Leverage)：能力检测可被多个模块使用
// - 可测试性：可以独立测试能力检测、纯计算路径、fallback 路径
// - 灵活性：用户可以注入自定义 fallback 策略
// ============================================================================

#include "tensor_core_capabilities.cuh"
#include "tensor_core_compute.cuh"
#include "tensor_core_launcher.cuh"
#include "tensor_core_launcher_impl.cuh"
