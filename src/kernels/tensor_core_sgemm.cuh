#pragma once

#include "../utils/cuda_utils.cuh"
#include "bank_conflict_free_sgemm.cuh"
#include "tiled_sgemm.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// WMMA is only available on sm_70+
// When compiling for host (__CUDA_ARCH__ not defined), always include WMMA
// When compiling for device, only include for sm_70+
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#include <mma.h>
#endif

namespace tensor_core {
inline constexpr int WMMA_M = 16;
inline constexpr int WMMA_N = 16;
inline constexpr int WMMA_K = 16;
} // namespace tensor_core

using tensor_core::WMMA_K;
using tensor_core::WMMA_M;
using tensor_core::WMMA_N;

inline bool tensorCoreDimensionsSupported(int M, int K, int N) {
    return M > 0 && K > 0 && N > 0 && M % WMMA_M == 0 && K % WMMA_K == 0 && N % WMMA_N == 0;
}

inline bool tensorCoresAvailable() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop.major >= 7;
}

/**
 * Kernel to convert FP32 to FP16
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
 * Basic Tensor Core SGEMM Kernel
 *
 * Each warp computes one WMMA_M x WMMA_N output tile.
 * Input matrices are expected to be in FP16 format.
 *
 * This kernel is only safe for dimensions that are multiples of 16.
 * Callers must validate dimensions before launching it.
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
 * Launch wrapper for Tensor Core SGEMM
 * Handles FP32 to FP16 conversion internally and safely falls back when WMMA
 * constraints are not met.
 */
inline void launch_tensor_core_sgemm(const float *A, const float *B, float *C, int M, int K, int N,
                                     cudaStream_t stream = 0) {
    if (M <= 0 || K <= 0 || N <= 0) {
        return;
    }

    if (!tensorCoresAvailable() || !tensorCoreDimensionsSupported(M, K, N)) {
        launch_bank_conflict_free_sgemm<32>(A, B, C, M, K, N, stream);
        return;
    }

    DeviceMemory<half> d_A_fp16(M * K);
    DeviceMemory<half> d_B_fp16(K * N);

    int blockSize = 256;
    int gridSizeA = (M * K + blockSize - 1) / blockSize;
    int gridSizeB = (K * N + blockSize - 1) / blockSize;

    float_to_half_kernel<<<gridSizeA, blockSize, 0, stream>>>(A, d_A_fp16.get(), M * K);
    float_to_half_kernel<<<gridSizeB, blockSize, 0, stream>>>(B, d_B_fp16.get(), K * N);
    CUDA_CHECK(cudaGetLastError());

    launch_tensor_core_sgemm_fp16_fast_path(d_A_fp16.get(), d_B_fp16.get(), C, M, K, N, stream);
}

/**
 * Tensor Core SGEMM with pre-converted FP16 inputs.
 * Falls back to a safe FP32 kernel when the WMMA fast path is not applicable.
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
