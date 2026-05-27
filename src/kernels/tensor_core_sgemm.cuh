#pragma once

// ============================================================================
// Tensor Core SGEMM - 深层模块
// ============================================================================
//
// 此模块提供完整的 Tensor Core SGEMM 功能，包括：
// - 设备能力查询
// - WMMA FP16→FP32 计算
// - FP32→FP16 类型转换
// - 统一启动接口（强制显式 fallback）
//
// 设计原则：
// - 深层模块：小接口，大实现
// - 不提供默认 fallback，强制调用者显式指定
// - 消除与具体内核的循环依赖
//
// 验证容差：
// - 使用 verify.cuh 中定义的 kTensorCoreVerifyTolerance
// ============================================================================

#include "../utils/cuda_utils.cuh"
#include "../utils/verify.cuh"
#include <climits>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <functional>

// ============================================================================
// WMMA Tile Dimensions
// ============================================================================

namespace tensor_core {
inline constexpr int WMMA_M = 16;
inline constexpr int WMMA_N = 16;
inline constexpr int WMMA_K = 16;
} // namespace tensor_core

using tensor_core::WMMA_K;
using tensor_core::WMMA_M;
using tensor_core::WMMA_N;

// ============================================================================
// Tensor Core Capabilities - 能力查询接口
// ============================================================================

/**
 * 检查当前设备是否支持 Tensor Core (sm_70+)
 *
 * 提供重载版本以支持可注入的 device info provider。
 */
inline bool tensorCoresAvailable(const DeviceInfoProvider &provider) {
    return provider.hasTensorCores();
}

/**
 * 检查当前设备是否支持 Tensor Core (默认使用生产环境设备)
 */
inline bool tensorCoresAvailable() {
    return tensorCoresAvailable(getProductionDeviceInfo());
}

/**
 * 检查给定维度是否适合 Tensor Core 加速
 * 所有维度必须是 WMMA tile 大小的倍数 (16)
 */
inline bool tensorCoreDimensionsSupported(int M, int K, int N) {
    return M > 0 && K > 0 && N > 0 && M % WMMA_M == 0 && K % WMMA_K == 0 && N % WMMA_N == 0;
}

/**
 * 获取当前设备的 Tensor Core 信息字符串
 *
 * 提供重载版本以支持可注入的 device info provider。
 */
inline const char *getTensorCoreArchName(const DeviceInfoProvider &provider) {
    int major = provider.computeMajor();
    int minor = provider.computeMinor();

    if (major == 7) {
        return (minor == 0) ? "Volta" : (minor == 5) ? "Turing" : "Unknown sm_7x";
    } else if (major == 8) {
        return (minor == 0 || minor == 6) ? "Ampere" : "Ampere/Ada";
    } else if (major == 9) {
        return "Hopper";
    }
    return "Unknown";
}

/**
 * 获取当前设备的 Tensor Core 信息字符串（默认使用生产环境设备）
 */
inline const char *getTensorCoreArchName() {
    return getTensorCoreArchName(getProductionDeviceInfo());
}

// ============================================================================
// Fallback 策略接口
// ============================================================================

/**
 * Fallback 内核函数类型
 *
 * 签名与所有标准 SGEMM 内核一致：
 * void(const float* A, const float* B, float* C, int M, int K, int N, cudaStream_t stream)
 */
using FallbackKernel =
    std::function<void(const float *, const float *, float *, int, int, int, cudaStream_t)>;

/**
 * 默认 fallback 策略
 *
 * 提供一个空的 fallback（用于测试或显式配置场景）
 */
[[maybe_unused]] inline void
nullFallback(const float *, const float *, float *, int, int, int, cudaStream_t = 0) {
    // 空实现 - 用于测试
}

// ============================================================================
// Tensor Core Compute - 纯 WMMA 计算路径
// ============================================================================

// WMMA is only available on sm_70+
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#include <mma.h>
#endif

/**
 * FP32 → FP16 转换内核
 */
__global__ void float_to_half_kernel(const float *__restrict__ input, half *__restrict__ output,
                                     int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

// WMMA kernel is only available on sm_70+
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

/**
 * 纯 Tensor Core WMMA 计算内核
 *
 * 每个 warp 计算一个 WMMA_M x WMMA_N 输出块。
 * 输入矩阵必须是 FP16 格式，维度必须是 16 的倍数。
 *
 * 注意：此内核不包含边界检查，调用者必须确保：
 * - 设备支持 sm_70+
 * - 所有维度都是 WMMA tile 大小的倍数
 */
__global__ void tensor_core_sgemm_kernel_fp16(const half *__restrict__ A,
                                              const half *__restrict__ B, float *__restrict__ C,
                                              int M, int K, int N) {
    int warpM = blockIdx.y;
    int warpN = blockIdx.x;

    int aRow = warpM * WMMA_M;
    int bCol = warpN * WMMA_N;

    if (aRow >= M || bCol >= N) {
        return;
    }

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           nvcuda::wmma::row_major>
        a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           nvcuda::wmma::row_major>
        b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        nvcuda::wmma::load_matrix_sync(a_frag, A + aRow * K + k, K);
        nvcuda::wmma::load_matrix_sync(b_frag, B + k * N + bCol, N);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    nvcuda::wmma::store_matrix_sync(C + aRow * N + bCol, c_frag, N, nvcuda::wmma::mem_row_major);
}

/**
 * FP16 计算路径的快速路径启动函数
 *
 * 前置条件：
 * - tensorCoresAvailable() == true
 * - tensorCoreDimensionsSupported(M, K, N) == true
 */
inline void launch_tensor_core_sgemm_fp16_fast_path(const half *A, const half *B, float *C, int M,
                                                    int K, int N, cudaStream_t stream = 0) {
    dim3 blockDim(kDefaultTileSize, 1);
    dim3 gridDim((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);

    tensor_core_sgemm_kernel_fp16<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, K, N);

    CUDA_CHECK(cudaGetLastError());
}

#else
// Stub implementations for older architectures (will not be called)
inline void launch_tensor_core_sgemm_fp16_fast_path(const half *, const half *, float *, int, int,
                                                    int, cudaStream_t) {
    // This function should never be called on pre-sm_70 GPUs
}
#endif

/**
 * 纯 WMMA 计算路径入口（FP16 输入）
 *
 * 此函数不执行 fallback，用于单独测试 Tensor Core 计算性能。
 * 如果设备或维度不支持，抛出异常。
 */
inline void launch_tensor_core_sgemm_fp16(const half *A, const half *B, float *C, int M, int K,
                                          int N, cudaStream_t stream = 0) {
    if (M <= 0 || K <= 0 || N <= 0) {
        return;
    }

    if (!tensorCoresAvailable() || !tensorCoreDimensionsSupported(M, K, N)) {
        throw CudaError("launch_tensor_core_sgemm_fp16 requires sm_70+ and dimensions aligned "
                        "to 16");
    }

    launch_tensor_core_sgemm_fp16_fast_path(A, B, C, M, K, N, stream);
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
inline void launch_tensor_core_sgemm_with_fallback(const float *A, const float *B, float *C, int M,
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

    int blockSize = kDefaultBlockSize;
    // 安全计算 gridSize，检查溢出
    auto safeGridSize = [](size_t num, int blk) -> int {
        size_t grid = (num + blk - 1) / blk;
        if (grid > static_cast<size_t>(INT_MAX)) {
            throw CudaError("Grid size overflow: matrix too large for kernel launch");
        }
        return static_cast<int>(grid);
    };
    int gridSizeA = safeGridSize(num_A, blockSize);
    int gridSizeB = safeGridSize(num_B, blockSize);

    // 检查矩阵元素数量是否超过 int 最大值
    if (num_A > static_cast<size_t>(INT_MAX)) {
        throw CudaError("Matrix A size overflow: too many elements for int parameter");
    }
    if (num_B > static_cast<size_t>(INT_MAX)) {
        throw CudaError("Matrix B size overflow: too many elements for int parameter");
    }

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
inline void launch_tensor_core_sgemm_with_fallback(const float *A, const float *B, float *C, int M,
                                                   int K, int N, const FallbackKernel &fallback,
                                                   cudaStream_t stream = 0) {
    launch_tensor_core_sgemm_with_fallback(A, B, C, M, K, N, fallback, stream);
}

// ============================================================================
// 注意：不提供默认 fallback 版本
// ============================================================================
//
// 此模块不提供 launch_tensor_core_sgemm() 默认入口。
// 调用者必须显式指定 fallback 策略：
//
//   launch_tensor_core_sgemm_with_fallback(A, B, C, M, K, N, myFallback, stream);
//
// 这种设计：
// - 消除与具体内核的循环依赖
// - 强制调用者思考"不支持时怎么办"
// - 保持模块职责单一（类型转换 + Tensor Core 启动）
