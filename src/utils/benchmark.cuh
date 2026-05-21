#pragma once

// ============================================================================
// SGEMM Benchmark 模块
//
// 此头文件聚合以下子模块：
// - benchmark_core.cuh: CUDA 事件计时、性能测量
// - benchmark_metrics.cuh: 指标计算、理论峰值
// - verify.cuh: cuBLAS 参考实现和验证
//
// 同时提供高级 Benchmark 类，简化常见使用场景。
// ============================================================================

#include "benchmark_core.cuh"
#include "benchmark_metrics.cuh"
#include "cuda_utils.cuh"
#include "verify.cuh"

#include <fstream>
#include <functional>
#include <string>
#include <vector>

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
    SGEMMBenchmark() { CUBLAS_CHECK(cublasCreate(&cublas_handle_)); }

    ~SGEMMBenchmark() {
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
        }
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
    BenchmarkResult run(const std::string &name, KernelFunc kernel_func, int M, int K, int N,
                        int warmup_runs = 5, int benchmark_runs = 20,
                        VerifyTolerance tolerance = kStandardVerifyTolerance) {
        BenchmarkResult result;
        result.kernel_name = name;
        result.M = M;
        result.K = K;
        result.N = N;

        // 初始化数据
        // 安全计算矩阵大小，避免整数溢出
        size_t size_A = static_cast<size_t>(M) * K;
        size_t size_B = static_cast<size_t>(K) * N;
        size_t size_C = static_cast<size_t>(M) * N;

        // 检查是否超过 size_t 范围（实际上是检查是否合理）
        if (size_A > static_cast<size_t>(INT_MAX) || size_B > static_cast<size_t>(INT_MAX) ||
            size_C > static_cast<size_t>(INT_MAX)) {
            throw CudaError("Matrix dimensions too large for benchmark");
        }

        std::vector<float> h_A(size_A), h_B(size_B), h_C(size_C), h_C_ref(size_C);
        DeviceMemory<float> d_A(size_A);
        DeviceMemory<float> d_B(size_B);
        DeviceMemory<float> d_C(size_C);
        DeviceMemory<float> d_C_ref(size_C);

        initRandomMatrix(h_A.data(), M, K, -1.0f, 1.0f, 42);
        initRandomMatrix(h_B.data(), K, N, -1.0f, 1.0f, 123);

        d_A.copyFromHost(h_A.data(), size_A);
        d_B.copyFromHost(h_B.data(), size_B);

        // 计算参考结果
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                 d_B.get(), N, d_A.get(), K, &beta, d_C_ref.get(), N));

        // 使用统一的 measureGpuTime 计时
        float time_ms = measureGpuTime([&]() { kernel_func(d_A.get(), d_B.get(), d_C.get(), M, K, N); },
                                       warmup_runs, benchmark_runs);

        // 计算指标
        PerformanceMetrics metrics = calculateSgemmMetrics(M, K, N, time_ms);
        result.time_ms = metrics.time_ms;
        result.gflops = metrics.gflops;
        result.bandwidth_gb_s = metrics.bandwidth_gb_s;
        result.efficiency = calculateEfficiency(result.gflops, getTheoreticalPeakGflops());

        // 验证正确性
        d_C.copyToHost(h_C.data(), size_C);
        d_C_ref.copyToHost(h_C_ref.data(), size_C);

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

        // 安全计算矩阵大小，避免整数溢出
        size_t size_A = static_cast<size_t>(M) * K;
        size_t size_B = static_cast<size_t>(K) * N;
        size_t size_C = static_cast<size_t>(M) * N;

        // 检查是否超过 size_t 范围
        if (size_A > static_cast<size_t>(INT_MAX) || size_B > static_cast<size_t>(INT_MAX) ||
            size_C > static_cast<size_t>(INT_MAX)) {
            throw CudaError("Matrix dimensions too large for benchmark");
        }

        std::vector<float> h_A(size_A), h_B(size_B);
        DeviceMemory<float> d_A(size_A);
        DeviceMemory<float> d_B(size_B);
        DeviceMemory<float> d_C(size_C);

        initRandomMatrix(h_A.data(), M, K, -1.0f, 1.0f, 42);
        initRandomMatrix(h_B.data(), K, N, -1.0f, 1.0f, 123);

        d_A.copyFromHost(h_A.data(), size_A);
        d_B.copyFromHost(h_B.data(), size_B);

        float alpha = 1.0f, beta = 0.0f;

        // 使用统一的 measureGpuTime 计时
        float time_ms = measureGpuTime(
            [&]() {
                CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                         d_B.get(), N, d_A.get(), K, &beta, d_C.get(), N));
            },
            warmup_runs, benchmark_runs);

        PerformanceMetrics metrics = calculateSgemmMetrics(M, K, N, time_ms);
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

        for (const auto &result : results_) {
            result.print();
        }

        printf("===================================================================="
               "============\n");
    }

    void exportRooflineData(const std::string &filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            fprintf(stderr, "Failed to open file: %s\n", filename.c_str());
            return;
        }

        file << "kernel,M,K,N,time_ms,gflops,bandwidth_gb_s,arithmetic_intensity\n";

        for (const auto &result : results_) {
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

    const std::vector<BenchmarkResult> &getResults() const { return results_; }
    void clearResults() { results_.clear(); }
    cublasHandle_t getCublasHandle() const { return cublas_handle_; }

  private:
    cublasHandle_t cublas_handle_;
    std::vector<BenchmarkResult> results_;
};

// ============================================================================
// 实用工具函数
// ============================================================================

inline void printPerformanceComparison(const std::vector<BenchmarkResult> &results,
                                       float cublas_gflops) {
    printf("\n");
    printf("Performance Comparison (vs cuBLAS):\n");
    printf("---------------------------------------------------------------------"
           "-----------\n");
    printf("  %-30s | %14s | %10s\n", "Kernel", "GFLOPS", "% of cuBLAS");
    printf("---------------------------------------------------------------------"
           "-----------\n");

    for (const auto &result : results) {
        float percentage = (result.gflops / cublas_gflops) * 100.0f;
        printf("  %-30s | %10.2f     | %8.1f%%\n", result.kernel_name.c_str(), result.gflops,
               percentage);
    }
    printf("---------------------------------------------------------------------"
           "-----------\n");
}
