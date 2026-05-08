#pragma once

// ============================================================================
// SGEMM Benchmark 模块
//
// 此头文件聚合以下子模块：
// - benchmark_core.cuh: CUDA 事件计时、性能测量
// - benchmark_metrics.cuh: 指标计算、理论峰值
// - benchmark_cublas.cuh: cuBLAS 参考实现
//
// 同时提供高级 Benchmark 类，简化常见使用场景。
// ============================================================================

#include "benchmark_core.cuh"
#include "benchmark_cublas.cuh"
#include "benchmark_metrics.cuh"
#include "cuda_utils.cuh"
#include "verify.cuh"

#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <string>
#include <vector>

// ============================================================================
// Benchmark 结果结构
// ============================================================================

struct BenchmarkResult {
    std::string kernel_name;
    int M, K, N;
    float time_ms;
    float gflops;
    float bandwidth_gb_s;
    bool correct;
    float max_error;
    float efficiency; // 理论峰值百分比

    void print() const {
        printf("  %-30s | %4d x %4d x %4d | %8.3f ms | %8.2f GFLOPS | %s | err: "
               "%.2e\n",
               kernel_name.c_str(), M, K, N, time_ms, gflops, correct ? "PASS" : "FAIL", max_error);
    }
};

// ============================================================================
// SGEMM Benchmark 类
// ============================================================================

/**
 * 高级 SGEMM Benchmark 编排器
 *
 * 提供完整的 benchmark 流程：
 * - 内核性能测量
 * - 正确性验证
 * - 结果汇总和导出
 */
class SGEMMBenchmark {
  public:
    SGEMMBenchmark() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~SGEMMBenchmark() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    /**
     * 运行内核 benchmark
     *
     * @tparam KernelFunc 内核函数类型
     * @param name 内核名称
     * @param kernel_func 内核启动函数
     * @param M, K, N 矩阵维度
     * @param warmup_runs 预热次数
     * @param benchmark_runs 计时次数
     * @param tolerance 验证容差
     */
    template <typename KernelFunc>
    BenchmarkResult run(const std::string& name, KernelFunc kernel_func, int M, int K, int N,
                        int warmup_runs = 5, int benchmark_runs = 20,
                        VerifyTolerance tolerance = kStandardVerifyTolerance) {
        BenchmarkResult result;
        result.kernel_name = name;
        result.M = M;
        result.K = K;
        result.N = N;

        // 初始化数据
        std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_C_ref(M * N);
        DeviceMemory<float> d_A(M * K);
        DeviceMemory<float> d_B(K * N);
        DeviceMemory<float> d_C(M * N);
        DeviceMemory<float> d_C_ref(M * N);

        initRandomMatrix(h_A.data(), M, K, -1.0f, 1.0f, 42);
        initRandomMatrix(h_B.data(), K, N, -1.0f, 1.0f, 123);

        d_A.copyFromHost(h_A.data(), M * K);
        d_B.copyFromHost(h_B.data(), K * N);

        // 计算参考结果
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                 d_B.get(), N, d_A.get(), K, &beta, d_C_ref.get(), N));

        // 预热和计时
        for (int i = 0; i < warmup_runs; ++i) {
            d_C.zero();
            kernel_func(d_A.get(), d_B.get(), d_C.get(), M, K, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start_));
        for (int i = 0; i < benchmark_runs; ++i) {
            kernel_func(d_A.get(), d_B.get(), d_C.get(), M, K, N);
        }
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));

        float total_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&total_time_ms, start_, stop_));

        // 计算指标
        PerformanceMetrics metrics = calculateSgemmMetrics(M, K, N, total_time_ms / benchmark_runs);
        result.time_ms = metrics.time_ms;
        result.gflops = metrics.gflops;
        result.bandwidth_gb_s = metrics.bandwidth_gb_s;
        result.efficiency = calculateEfficiency(result.gflops, getTheoreticalPeakGflops());

        // 验证正确性
        d_C.copyToHost(h_C.data(), M * N);
        d_C_ref.copyToHost(h_C_ref.data(), M * N);

        VerifyResult verify_result = compareMatrices(h_C.data(), h_C_ref.data(), M, N, tolerance);
        result.correct = verify_result.passed;
        result.max_error = verify_result.max_rel_error;

        results_.push_back(result);
        return result;
    }

    /**
     * 测量 cuBLAS 性能作为基线
     */
    BenchmarkResult runCublas(int M, int K, int N, int warmup_runs = 5, int benchmark_runs = 20) {
        BenchmarkResult result;
        result.kernel_name = "cuBLAS";
        result.M = M;
        result.K = K;
        result.N = N;

        std::vector<float> h_A(M * K), h_B(K * N);
        DeviceMemory<float> d_A(M * K);
        DeviceMemory<float> d_B(K * N);
        DeviceMemory<float> d_C(M * N);

        initRandomMatrix(h_A.data(), M, K, -1.0f, 1.0f, 42);
        initRandomMatrix(h_B.data(), K, N, -1.0f, 1.0f, 123);

        d_A.copyFromHost(h_A.data(), M * K);
        d_B.copyFromHost(h_B.data(), K * N);

        float alpha = 1.0f, beta = 0.0f;

        for (int i = 0; i < warmup_runs; ++i) {
            CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                     d_B.get(), N, d_A.get(), K, &beta, d_C.get(), N));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start_));
        for (int i = 0; i < benchmark_runs; ++i) {
            CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                     d_B.get(), N, d_A.get(), K, &beta, d_C.get(), N));
        }
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));

        float total_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&total_time_ms, start_, stop_));

        PerformanceMetrics metrics = calculateSgemmMetrics(M, K, N, total_time_ms / benchmark_runs);
        result.time_ms = metrics.time_ms;
        result.gflops = metrics.gflops;
        result.bandwidth_gb_s = metrics.bandwidth_gb_s;
        result.efficiency = calculateEfficiency(result.gflops, getTheoreticalPeakGflops());

        result.correct = true;
        result.max_error = 0.0f;
        results_.push_back(result);
        return result;
    }

    void printSummary() const {
        printf("\n");
        printf("===================================================================="
               "============\n");
        printf("                           SGEMM Benchmark Results\n");
        printf("===================================================================="
               "============\n");
        printf("  %-30s | %-17s | %10s | %14s | %4s | %s\n", "Kernel", "Dimensions", "Time",
               "Performance", "Pass", "Max Error");
        printf("--------------------------------------------------------------------"
               "------------\n");

        for (const auto& result : results_) {
            result.print();
        }

        printf("===================================================================="
               "============\n");
    }

    void exportRooflineData(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            fprintf(stderr, "Failed to open file: %s\n", filename.c_str());
            return;
        }

        file << "kernel,M,K,N,time_ms,gflops,bandwidth_gb_s,arithmetic_intensity\n";

        for (const auto& result : results_) {
            double flops = 2.0 * result.M * result.N * result.K;
            double bytes =
                (result.M * result.K + result.K * result.N + result.M * result.N) * sizeof(float);
            double ai = flops / bytes;

            file << result.kernel_name << "," << result.M << "," << result.K << "," << result.N
                 << "," << result.time_ms << "," << result.gflops << "," << result.bandwidth_gb_s
                 << "," << ai << "\n";
        }

        file.close();
        printf("Approximate roofline data exported to: %s\n", filename.c_str());
    }

    const std::vector<BenchmarkResult>& getResults() const { return results_; }
    void clearResults() { results_.clear(); }
    cublasHandle_t getCublasHandle() const { return cublas_handle_; }

  private:
    cudaEvent_t start_, stop_;
    cublasHandle_t cublas_handle_;
    std::vector<BenchmarkResult> results_;
};

// ============================================================================
// 实用工具函数
// ============================================================================

inline void printPerformanceComparison(const std::vector<BenchmarkResult>& results,
                                       float cublas_gflops) {
    printf("\n");
    printf("Performance Comparison (vs cuBLAS):\n");
    printf("---------------------------------------------------------------------"
           "-----------\n");
    printf("  %-30s | %14s | %10s\n", "Kernel", "GFLOPS", "% of cuBLAS");
    printf("---------------------------------------------------------------------"
           "-----------\n");

    for (const auto& result : results) {
        float percentage = (result.gflops / cublas_gflops) * 100.0f;
        printf("  %-30s | %10.2f     | %8.1f%%\n", result.kernel_name.c_str(), result.gflops,
               percentage);
    }
    printf("---------------------------------------------------------------------"
           "-----------\n");
}
