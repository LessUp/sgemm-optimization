#pragma once

// ============================================================================
// Tensor Core 默认 Fallback 策略
// ============================================================================
//
// 此文件提供 Tensor Core 的默认 fallback 实现。
// 独立于 tensor_core_sgemm.cuh 以避免循环依赖。
// ============================================================================

#include "bank_conflict_free_sgemm.cuh"
#include "tensor_core_sgemm.cuh"

/**
 * 默认 Tensor Core fallback 策略
 *
 * 使用 bank-conflict-free 内核作为 fallback。
 * 这是一个架构级决策：当 Tensor Core 不可用时，使用最优的 FP32 内核。
 */
struct BankConflictFreeFallback {
    void operator()(const float *A, const float *B, float *C, int M, int K, int N,
                    cudaStream_t stream = 0) const {
        launch_bank_conflict_free_sgemm<32>(A, B, C, M, K, N, stream);
    }
};

/**
 * 便利函数：返回默认 fallback 策略
 *
 * 用于兼容现有代码中的 lambda 表达式。
 */
inline auto defaultTensorCoreFallback() {
    return BankConflictFreeFallback{};
}
