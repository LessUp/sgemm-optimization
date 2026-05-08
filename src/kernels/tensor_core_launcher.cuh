#pragma once

#include "../utils/cuda_utils.cuh"
#include "tensor_core_capabilities.cuh"
#include "tensor_core_compute.cuh"
#include <cuda_runtime.h>
#include <functional>

// ============================================================================
// Tensor Core Verification Tolerance
// ============================================================================

// Tensor Core 使用 FP16 中间精度，需要更宽松的容差
// 此容差定义在 Tensor Core 模块中，保持精度相关常量与其实现在一起
inline constexpr VerifyTolerance kTensorCoreVerifyTolerance{5e-2f, 1e-2f};

// ============================================================================
// Fallback 策略接口
// ============================================================================

/**
 * Fallback 内核函数类型
 *
 * 签名与所有标准 SGEMM 内核一致：
 * void(const float* A, const float* B, float* C, int M, int K, int N, cudaStream_t stream)
 */
using FallbackKernel = std::function<void(const float*, const float*, float*, int, int, int,
                                          cudaStream_t)>;

/**
 * 默认 fallback 策略
 *
 * 提供一个空的 fallback（用于测试或显式配置场景）
 */
inline void nullFallback(const float*, const float*, float*, int, int, int, cudaStream_t = 0) {
    // 空实现 - 用于测试
}

// ============================================================================
// Tensor Core Launcher - 统一启动接口（支持自定义 fallback）
// ============================================================================

/**
 * 安全的端到端 Tensor Core SGEMM 启动器（模板版本）
 *
 * 此函数是 Tensor Core 的公共 FP32 入口点：
 * - 在支持的设备上使用 WMMA 计算
 * - 自动处理 FP32 → FP16 类型转换
 * - 不支持的设备或维度使用指定的 fallback 策略
 *
 * @tparam FallbackFunc fallback 函数类型（可调用对象）
 * @param A FP32 输入矩阵 A (M x K)
 * @param B FP32 输入矩阵 B (K x N)
 * @param C FP32 输出矩阵 C (M x N)
 * @param M, K, N 矩阵维度
 * @param fallback fallback 内核函数（当 Tensor Core 不可用时）
 * @param stream CUDA 流
 */
template <typename FallbackFunc>
inline void launch_tensor_core_sgemm_with_fallback(const float* A, const float* B, float* C, int M,
                                                   int K, int N, FallbackFunc fallback,
                                                   cudaStream_t stream = 0) {
    if (M <= 0 || K <= 0 || N <= 0) {
        return;
    }

    // Fallback 路径：设备或维度不支持 Tensor Core
    if (!tensorCoresAvailable() || !tensorCoreDimensionsSupported(M, K, N)) {
        fallback(A, B, C, M, K, N, stream);
        return;
    }

    // Tensor Core 快速路径：转换 FP32 → FP16 并执行 WMMA
    size_t num_A = static_cast<size_t>(M) * K;
    size_t num_B = static_cast<size_t>(K) * N;
    DeviceMemory<half> d_A_fp16(num_A);
    DeviceMemory<half> d_B_fp16(num_B);

    int blockSize = 256;
    int gridSizeA = static_cast<int>((num_A + blockSize - 1) / blockSize);
    int gridSizeB = static_cast<int>((num_B + blockSize - 1) / blockSize);

    float_to_half_kernel<<<gridSizeA, blockSize, 0, stream>>>(A, d_A_fp16.get(),
                                                              static_cast<int>(num_A));
    float_to_half_kernel<<<gridSizeB, blockSize, 0, stream>>>(B, d_B_fp16.get(),
                                                              static_cast<int>(num_B));
    CUDA_CHECK(cudaGetLastError());

    launch_tensor_core_sgemm_fp16_fast_path(d_A_fp16.get(), d_B_fp16.get(), C, M, K, N, stream);
}

/**
 * 重载版本：使用 std::function 作为 fallback
 *
 * 允许运行时选择 fallback 策略。
 */
inline void launch_tensor_core_sgemm_with_fallback(const float* A, const float* B, float* C, int M,
                                                   int K, int N, const FallbackKernel& fallback,
                                                   cudaStream_t stream = 0) {
    launch_tensor_core_sgemm_with_fallback(A, B, C, M, K, N, fallback, stream);
}

// ============================================================================
// 默认接口声明
// ============================================================================

/**
 * 使用 bank-conflict-free 作为默认 fallback
 *
 * 注意：此函数实现在 tensor_core_launcher_impl.cuh 中，
 * 以避免与 bank_conflict_free_sgemm.cuh 的循环依赖。
 *
 * 使用方法：
 * 1. 包含 tensor_core_launcher.cuh 和 bank_conflict_free_sgemm.cuh
 * 2. 直接调用 launch_tensor_core_sgemm()
 */
inline void launch_tensor_core_sgemm(const float* A, const float* B, float* C, int M, int K, int N,
                                     cudaStream_t stream = 0);
