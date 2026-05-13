#pragma once

#include "../utils/cuda_utils.cuh"
#include <cuda_runtime.h>

// WMMA tile dimensions
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
 */
inline bool tensorCoresAvailable() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop.major >= 7;
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
 */
inline const char *getTensorCoreArchName() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    if (prop.major == 7) {
        return (prop.minor == 0) ? "Volta" : (prop.minor == 5) ? "Turing" : "Unknown sm_7x";
    } else if (prop.major == 8) {
        return (prop.minor == 0 || prop.minor == 6) ? "Ampere" : "Ampere/Ada";
    } else if (prop.major == 9) {
        return "Hopper";
    }
    return "Unknown";
}
