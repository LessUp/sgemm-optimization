#pragma once

#include "benchmark_core.cuh"
#include "benchmark_metrics.cuh"
#include "cuda_utils.cuh"
#include "verify.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

// ============================================================================
// cuBLAS SGEMM 参考实现
// ============================================================================

/**
 * cuBLAS SGEMM 参考调用器
 *
 * 封装 cuBLAS 句柄管理和 SGEMM 调用，提供：
 * - 参考结果计算
 * - 性能测量
 * - 自动句柄生命周期管理
 */
class CublasSgemm {
  public:
    CublasSgemm() { CUBLAS_CHECK(cublasCreate(&handle_)); }

    ~CublasSgemm() {
        if (handle_) {
            cublasDestroy(handle_);
        }
    }

    // 禁用拷贝
    CublasSgemm(const CublasSgemm &) = delete;
    CublasSgemm &operator=(const CublasSgemm &) = delete;

    /**
     * 执行 SGEMM: C = alpha * A * B + beta * C
     *
     * @param A M x K 矩阵（行优先）
     * @param B K x N 矩阵（行优先）
     * @param C M x N 矩阵（行优先）
     *
     * 注意：cuBLAS 使用列优先，内部进行转置处理
     */
    void sgemm(const float *d_A, const float *d_B, float *d_C, int M, int K, int N,
               float alpha = 1.0f, float beta = 0.0f) {
        // cuBLAS 使用列优先，计算 C^T = B^T * A^T
        // 结果 C 仍为行优先格式
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K,
                                 &beta, d_C, N));
    }

    /**
     * 测量 cuBLAS SGEMM 性能
     *
     * @return 平均执行时间（毫秒）
     */
    float measurePerformance(const float *d_A, const float *d_B, float *d_C, int M, int K, int N,
                             int warmup_runs = 5, int benchmark_runs = 20) {
        float alpha = 1.0f, beta = 0.0f;

        auto run_func = [&]() {
            CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N,
                                     d_A, K, &beta, d_C, N));
        };

        return measureGpuTime(run_func, warmup_runs, benchmark_runs);
    }

    cublasHandle_t getHandle() const { return handle_; }

  private:
    cublasHandle_t handle_;
};
