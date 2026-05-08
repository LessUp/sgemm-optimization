#pragma once

#include "../utils/cuda_utils.cuh"
#include "tensor_core_capabilities.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// WMMA is only available on sm_70+
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#include <mma.h>
#endif

// ============================================================================
// Tensor Core Compute - 纯 WMMA 计算路径
// ============================================================================

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
    dim3 blockDim(32, 1);
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
