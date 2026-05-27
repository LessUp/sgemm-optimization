#pragma once

#include "cli_parser.cuh"
#include "kernels/kernel_catalog.cuh"
#include "kernels/tensor_core_benchmark.cuh"
#include "kernels/tensor_core_fallback.cuh"
#include "kernels/tensor_core_sgemm.cuh"
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
        BenchmarkResult cublas_result = benchmark.runCublas(
            M, K, N, config_.settings.run.warmup_runs, config_.settings.run.benchmark_runs);
        float cublas_gflops = cublas_result.gflops;

        // 标准内核
        runStandardKernels(benchmark, M, K, N);

        // Tensor Core 内核
        runTensorCoreKernels(benchmark, M, K, N);

        // 报告
        benchmark.printSummary();
        printPerformanceComparison(benchmark.getResults(), cublas_gflops);

        // 导出 roofline 数据 (仅当设置启用时)
        if (config_.settings.output.export_roofline) {
            std::string filename = config_.settings.output.makeRooflineFilename(M, K, N);
            benchmark.exportRooflineData(filename);
        }
    }

    void runStandardKernels(SGEMMBenchmark &benchmark, int M, int K, int N) {
        const auto& catalog = getKernelCatalog();
        
        for (const auto& entry : catalog) {
            if (entry.type != KernelType::Standard) {
                continue;
            }
            
            VerifyTolerance tolerance = config_.settings.toleranceForKernel(entry.type);
            
            printf("Running %s...\n", entry.name.c_str());
            benchmark.run(
                entry.name,
                entry.launcher,
                M, K, N,
                config_.settings.run.warmup_runs,
                config_.settings.run.benchmark_runs,
                tolerance);
        }
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

        if (!tensorCoreDimensionsSupported(M, K, N)) {
            printf("Skipping Tensor Core benchmarks for %d x %d x %d "
                   "(requires positive dimensions aligned to 16, fallback would mislabel FP32 as WMMA).\n",
                   M, K, N);
            return;
        }

        // Run catalog-based tensor core kernels
        const auto& catalog = getKernelCatalog();
        for (const auto& entry : catalog) {
            if (entry.type != KernelType::TensorCore) {
                continue;
            }
            
            VerifyTolerance tolerance = config_.settings.toleranceForKernel(entry.type);
            
            printf("Running %s, includes FP32->FP16 "
                   "conversion/fallback)...\n", entry.name.c_str());
            benchmark.run(
                entry.name,
                entry.launcher,
                M, K, N,
                config_.settings.run.warmup_runs,
                config_.settings.run.benchmark_runs,
                tolerance);
        }

        // Tensor Core compute-only benchmark remains special-cased
        // (requires cublas handle, different interface)
        VerifyTolerance tolerance = config_.settings.toleranceForKernel(KernelType::TensorCore);
        printf("Running Tensor Core SGEMM (compute-only WMMA path)...\n");
        BenchmarkResult tc_result = runTensorCoreComputeOnlyBenchmark(
            benchmark.getCublasHandle(), M, K, N, config_.settings.run.warmup_runs,
            config_.settings.run.benchmark_runs, tolerance);
        tc_result.print();
    }

    BenchmarkConfig config_;
};
