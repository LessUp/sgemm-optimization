#pragma once

#include "bank_conflict_free_sgemm.cuh"
#include "tensor_core_launcher.cuh"

// ============================================================================
// 默认 launch_tensor_core_sgemm 实现
// ============================================================================

/**
 * 使用 bank-conflict-free 内核作为默认 fallback
 *
 * 这是最常用的配置，提供：
 * - Tensor Core 加速（当可用且维度对齐时）
 * - Bank-conflict-free fallback（当 Tensor Core 不可用时）
 */
inline void launch_tensor_core_sgemm(const float* A, const float* B, float* C, int M, int K, int N,
                                     cudaStream_t stream) {
    auto default_fallback = [](const float* A, const float* B, float* C, int M, int K, int N,
                               cudaStream_t s) {
        launch_bank_conflict_free_sgemm<32>(A, B, C, M, K, N, s);
    };

    launch_tensor_core_sgemm_with_fallback(A, B, C, M, K, N, default_fallback, stream);
}
