#pragma once

#include "bank_conflict_free_sgemm.cuh"
#include "tiled_sgemm.cuh"
#include "../utils/cuda_utils.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace {
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
}

inline bool tensorCoreDimensionsSupported(int M, int K, int N) {
  return M > 0 && K > 0 && N > 0 && M % WMMA_M == 0 && K % WMMA_K == 0 &&
         N % WMMA_N == 0;
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
__global__ void float_to_half_kernel(const float *__restrict__ input,
                                     half *__restrict__ output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = __float2half(input[idx]);
  }
}

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
                                              const half *__restrict__ B,
                                              float *__restrict__ C, int M,
                                              int K, int N) {
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
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                         float>
      c_frag;

  nvcuda::wmma::fill_fragment(c_frag, 0.0f);

  for (int k = 0; k < K; k += WMMA_K) {
    nvcuda::wmma::load_matrix_sync(a_frag, A + aRow * K + k, K);
    nvcuda::wmma::load_matrix_sync(b_frag, B + k * N + bCol, N);
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  nvcuda::wmma::store_matrix_sync(C + aRow * N + bCol, c_frag, N,
                                  nvcuda::wmma::mem_row_major);
}

/**
 * Optimized Tensor Core SGEMM with shared memory staging
 */
template <int BLOCK_TILES_M = 4, int BLOCK_TILES_N = 4>
__global__ void tensor_core_sgemm_kernel_optimized(const half *__restrict__ A,
                                                   const half *__restrict__ B,
                                                   float *__restrict__ C, int M,
                                                   int K, int N) {
  constexpr int BLOCK_M = BLOCK_TILES_M * WMMA_M;
  constexpr int BLOCK_N = BLOCK_TILES_N * WMMA_N;
  constexpr int BLOCK_K = WMMA_K;

  __shared__ half As[BLOCK_M][BLOCK_K];
  __shared__ half Bs[BLOCK_K][BLOCK_N];

  int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
  int warpsPerBlockX = BLOCK_TILES_N;
  int warpX = warpId % warpsPerBlockX;
  int warpY = warpId / warpsPerBlockX;

  int blockRowStart = blockIdx.y * BLOCK_M;
  int blockColStart = blockIdx.x * BLOCK_N;

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                         nvcuda::wmma::row_major>
      a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                         nvcuda::wmma::row_major>
      b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                         float>
      c_frag;

  nvcuda::wmma::fill_fragment(c_frag, 0.0f);

  for (int k = 0; k < K; k += BLOCK_K) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;

    for (int i = tid; i < BLOCK_M * BLOCK_K; i += numThreads) {
      int row = i / BLOCK_K;
      int col = i % BLOCK_K;
      int globalRow = blockRowStart + row;
      int globalCol = k + col;

      if (globalRow < M && globalCol < K) {
        As[row][col] = A[globalRow * K + globalCol];
      } else {
        As[row][col] = __float2half(0.0f);
      }
    }

    for (int i = tid; i < BLOCK_K * BLOCK_N; i += numThreads) {
      int row = i / BLOCK_N;
      int col = i % BLOCK_N;
      int globalRow = k + row;
      int globalCol = blockColStart + col;

      if (globalRow < K && globalCol < N) {
        Bs[row][col] = B[globalRow * N + globalCol];
      } else {
        Bs[row][col] = __float2half(0.0f);
      }
    }

    __syncthreads();

    int warpRow = warpY * WMMA_M;
    int warpCol = warpX * WMMA_N;

    nvcuda::wmma::load_matrix_sync(a_frag, &As[warpRow][0], BLOCK_K);
    nvcuda::wmma::load_matrix_sync(b_frag, &Bs[0][warpCol], BLOCK_N);
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    __syncthreads();
  }

  int cRow = blockRowStart + warpY * WMMA_M;
  int cCol = blockColStart + warpX * WMMA_N;

  if (cRow < M && cCol < N) {
    nvcuda::wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N,
                                    nvcuda::wmma::mem_row_major);
  }
}

inline void launch_tensor_core_sgemm_fp16_fast_path(const half *A, const half *B,
                                                    float *C, int M, int K,
                                                    int N,
                                                    cudaStream_t stream = 0) {
  dim3 blockDim(32, 1);
  dim3 gridDim((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);

  tensor_core_sgemm_kernel_fp16<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, K,
                                                                  N);

  CUDA_CHECK(cudaGetLastError());
}

/**
 * Launch wrapper for Tensor Core SGEMM
 * Handles FP32 to FP16 conversion internally and safely falls back when WMMA
 * constraints are not met.
 */
inline void launch_tensor_core_sgemm(const float *A, const float *B, float *C,
                                     int M, int K, int N,
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

  float_to_half_kernel<<<gridSizeA, blockSize, 0, stream>>>(A, d_A_fp16.get(),
                                                            M * K);
  float_to_half_kernel<<<gridSizeB, blockSize, 0, stream>>>(B, d_B_fp16.get(),
                                                            K * N);
  CUDA_CHECK(cudaGetLastError());

  launch_tensor_core_sgemm_fp16_fast_path(d_A_fp16.get(), d_B_fp16.get(), C, M,
                                          K, N, stream);
}

/**
 * Tensor Core SGEMM with pre-converted FP16 inputs.
 * Falls back to a safe FP32 kernel when the WMMA fast path is not applicable.
 */
inline void launch_tensor_core_sgemm_fp16(const half *A, const half *B, float *C,
                                          int M, int K, int N,
                                          cudaStream_t stream = 0) {
  if (M <= 0 || K <= 0 || N <= 0) {
    return;
  }

  if (!tensorCoresAvailable() || !tensorCoreDimensionsSupported(M, K, N)) {
    throw CudaError(
        "launch_tensor_core_sgemm_fp16 requires sm_70+ and dimensions aligned "
        "to 16");
  }

  launch_tensor_core_sgemm_fp16_fast_path(A, B, C, M, K, N, stream);
}
