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
    CublasSgemm(const CublasSgemm&) = delete;
    CublasSgemm& operator=(const CublasSgemm&) = delete;

    /**
     * 执行 SGEMM: C = alpha * A * B + beta * C
     *
     * @param A M x K 矩阵（行优先）
     * @param B K x N 矩阵（行优先）
     * @param C M x N 矩阵（行优先）
     *
     * 注意：cuBLAS 使用列优先，内部进行转置处理
     */
    void sgemm(const float* d_A, const float* d_B, float* d_C, int M, int K, int N,
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
    float measurePerformance(const float* d_A, const float* d_B, float* d_C, int M, int K, int N,
                             int warmup_runs = 5, int benchmark_runs = 20) {
        float alpha = 1.0f, beta = 0.0f;

        auto run_func = [&]() {
            CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A,
                                     K, &beta, d_C, N));
        };

        return measureGpuTime(run_func, warmup_runs, benchmark_runs);
    }

    cublasHandle_t getHandle() const { return handle_; }

  private:
    cublasHandle_t handle_;
};

// ============================================================================
// SGEMM 参考计算器
// ============================================================================

/**
 * 计算参考 SGEMM 结果并验证
 *
 * 提供完整的参考计算流程：
 * - 设备内存分配
 * - 随机矩阵初始化
 * - cuBLAS 参考计算
 * - 结果验证
 */
class SgemmReferenceCalculator {
  public:
    SgemmReferenceCalculator() = default;

    /**
     * 初始化参考数据
     *
     * @param M, K, N 矩阵维度
     * @param seed 随机种子（可重现）
     */
    void initialize(int M, int K, int N, unsigned int seed_A = 42, unsigned int seed_B = 123) {
        M_ = M;
        K_ = K;
        N_ = N;

        h_A_.resize(M * K);
        h_B_.resize(K * N);

        initRandomMatrix(h_A_.data(), M, K, -1.0f, 1.0f, seed_A);
        initRandomMatrix(h_B_.data(), K, N, -1.0f, 1.0f, seed_B);

        d_A_ = std::make_unique<DeviceMemory<float>>(M * K);
        d_B_ = std::make_unique<DeviceMemory<float>>(K * N);
        d_C_ref_ = std::make_unique<DeviceMemory<float>>(M * N);

        d_A_->copyFromHost(h_A_.data(), M * K);
        d_B_->copyFromHost(h_B_.data(), K * N);
    }

    /**
     * 计算参考结果
     */
    void computeReference() {
        cublas_.sgemm(d_A_->get(), d_B_->get(), d_C_ref_->get(), M_, K_, N_);
    }

    /**
     * 获取主机端参考结果
     */
    std::vector<float> getReferenceResult() {
        std::vector<float> h_ref(M_ * N_);
        d_C_ref_->copyToHost(h_ref.data(), M_ * N_);
        return h_ref;
    }

    // 访问器
    const std::vector<float>& hostA() const { return h_A_; }
    const std::vector<float>& hostB() const { return h_B_; }
    float* deviceA() { return d_A_->get(); }
    float* deviceB() { return d_B_->get(); }
    float* deviceCRef() { return d_C_ref_->get(); }
    CublasSgemm& cublas() { return cublas_; }
    int M() const { return M_; }
    int K() const { return K_; }
    int N() const { return N_; }

  private:
    int M_ = 0, K_ = 0, N_ = 0;
    std::vector<float> h_A_, h_B_;
    std::unique_ptr<DeviceMemory<float>> d_A_;
    std::unique_ptr<DeviceMemory<float>> d_B_;
    std::unique_ptr<DeviceMemory<float>> d_C_ref_;
    CublasSgemm cublas_;
};
