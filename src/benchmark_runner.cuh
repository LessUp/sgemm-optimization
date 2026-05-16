#pragma once

#include "cli_parser.cuh"
#include "kernels/double_buffer_sgemm.cuh"
#include "kernels/naive_sgemm.cuh"
#include "kernels/tensor_core_benchmark.cuh"
#include "kernels/tensor_core_fallback.cuh"
#include "kernels/tensor_core_sgemm.cuh"
#include "kernels/tiled_sgemm.cuh"
#include "utils/benchmark.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/verify.cuh"

#include <cstdio>

// ============================================================================
// Benchmark 编排器
// ============================================================================

/**
 * SGEMM Benchmark 编排器
 *
 * 负责调度所有内核 benchmark 并生成报告。
 * 与 CLI 解析分离，可被测试或脚本直接调用。
 */
class BenchmarkRunner {
  public:
    explicit BenchmarkRunner(const BenchmarkConfig &config) : config_(config) {}

    /**
     * 运行所有配置的 benchmark
     */
    void runAll() {
        printHeader();

        for (const auto &[M, K, N] : config_.dimensions) {
            runBenchmarks(M, K, N);
        }

        printFooter();
    }

  private:
    void printHeader() const {
        printf("\n");
        printf("====================================================================="
               "===========\n");
        printf("                    SGEMM Optimization Benchmark Suite\n");
        printf("====================================================================="
               "===========\n");

        printGPUInfo();

        float peakGflops = getTheoreticalPeakGflops();
        float peakBandwidth = getTheoreticalPeakBandwidth();
        printf("Approximate theoretical peak FP32: %.2f GFLOPS\n", peakGflops);
        printf("Approximate theoretical peak bandwidth: %.2f GB/s\n", peakBandwidth);
        printf("\n");
    }

    void printFooter() const {
        printf("\n");
        printf("====================================================================="
               "===========\n");
        printf("                           Benchmark Complete\n");
        printf("====================================================================="
               "===========\n");
        printf("\n");
        printf("Notes:\n");
        printf("  - Standard kernels are verified with shared FP32 tolerances.\n");
        printf("  - Tensor Core verification uses relaxed mixed-precision tolerances.\n");
        printf("  - The end-to-end Tensor Core result includes FP32->FP16 conversion "
               "and safe fallback behavior.\n");
        printf("  - The compute-only Tensor Core result is only shown for "
               "WMMA-compatible dimensions.\n");
        printf("\n");
    }

    void runBenchmarks(int M, int K, int N) {
        printf("\n");
        printf("====================================================================="
               "===========\n");
        printf("                    Benchmarking %d x %d x %d SGEMM\n", M, K, N);
        printf("====================================================================="
               "===========\n");

        SGEMMBenchmark benchmark;

        // cuBLAS 参考
        printf("\nRunning cuBLAS (reference)...\n");
        BenchmarkResult cublas_result =
            benchmark.runCublas(M, K, N, config_.warmup_runs, config_.benchmark_runs);
        float cublas_gflops = cublas_result.gflops;

        // 标准内核
        runStandardKernels(benchmark, M, K, N);

        // Tensor Core 内核
        runTensorCoreKernels(benchmark, M, K, N);

        // 报告
        benchmark.printSummary();
        printPerformanceComparison(benchmark.getResults(), cublas_gflops);

        // 导出 roofline 数据
        char filename[kFilenameBufferSize];
        snprintf(filename, sizeof(filename), "roofline_data_%d_%d_%d.csv", M, K, N);
        benchmark.exportRooflineData(filename);
    }

    void runStandardKernels(SGEMMBenchmark &benchmark, int M, int K, int N) {
        printf("Running Naive SGEMM...\n");
        benchmark.run(
            "Naive",
            [](const float *A, const float *B, float *C, int M, int K, int N) {
                launch_naive_sgemm<32>(A, B, C, M, K, N);
            },
            M, K, N, config_.warmup_runs, config_.benchmark_runs, kStandardVerifyTolerance);

        printf("Running Tiled SGEMM...\n");
        benchmark.run(
            "Tiled (32x32)",
            [](const float *A, const float *B, float *C, int M, int K, int N) {
                launch_tiled_sgemm<32>(A, B, C, M, K, N);
            },
            M, K, N, config_.warmup_runs, config_.benchmark_runs, kStandardVerifyTolerance);

        printf("Running Bank Conflict Free SGEMM...\n");
        benchmark.run(
            "Bank Conflict Free",
            [](const float *A, const float *B, float *C, int M, int K, int N) {
                launch_bank_conflict_free_sgemm<32>(A, B, C, M, K, N);
            },
            M, K, N, config_.warmup_runs, config_.benchmark_runs, kStandardVerifyTolerance);

        printf("Running Double Buffer SGEMM...\n");
        benchmark.run(
            "Double Buffer",
            [](const float *A, const float *B, float *C, int M, int K, int N) {
                launch_double_buffer_sgemm<32>(A, B, C, M, K, N);
            },
            M, K, N, config_.warmup_runs, config_.benchmark_runs, kStandardVerifyTolerance);
    }

    void runTensorCoreKernels(SGEMMBenchmark &benchmark, int M, int K, int N) {
        if (!tensorCoresAvailable()) {
            int device;
            CUDA_CHECK(cudaGetDevice(&device));
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
            printf("Skipping Tensor Core benchmarks (requires sm_70+, current: sm_%d%d)\n",
                   prop.major, prop.minor);
            return;
        }

        printf("Running Tensor Core SGEMM (end-to-end, includes FP32->FP16 "
               "conversion/fallback)...\n");
        benchmark.run(
            "Tensor Core (WMMA end-to-end)",
            [](const float *A, const float *B, float *C, int M, int K, int N) {
                launch_tensor_core_sgemm_with_fallback(A, B, C, M, K, N,
                                                       defaultTensorCoreFallback());
            },
            M, K, N, config_.warmup_runs, config_.benchmark_runs, kTensorCoreVerifyTolerance);

        if (tensorCoreDimensionsSupported(M, K, N)) {
            printf("Running Tensor Core SGEMM (compute-only WMMA path)...\n");
            BenchmarkResult tc_result = runTensorCoreComputeOnlyBenchmark(
                benchmark.getCublasHandle(), M, K, N, config_.warmup_runs, config_.benchmark_runs,
                kTensorCoreVerifyTolerance);
            tc_result.print();
        } else {
            printf("Skipping Tensor Core compute-only benchmark (requires positive "
                   "dimensions aligned to 16).\n");
        }
    }

    BenchmarkConfig config_;
};
