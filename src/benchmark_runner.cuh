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
 *
 * 设计：
 * - 使用 KernelCatalog 作为内核阶梯的唯一事实来源
 * - Catalog 条目包含所有约束和默认容差
 * - BenchmarkRunner 只负责编排流程，不包含内核特定逻辑
 */
class BenchmarkRunner {
  public:
    explicit BenchmarkRunner(const BenchmarkConfig &config) : config_(config) {}

    /**
     * 运行所有配置的 benchmark
     */
    bool runAll() {
        printHeader();

        if (!cudaDeviceAvailable()) {
            fprintf(stderr,
                    "No CUDA-capable device detected. Benchmark execution requires a local CUDA "
                    "environment with a visible GPU.\n");
            return false;
        }

        for (const auto &[M, K, N] : config_.dimensions) {
            runBenchmarks(M, K, N);
        }

        printFooter();
        return true;
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

        if (cudaDeviceAvailable()) {
            float peakGflops = getTheoreticalPeakGflops();
            float peakBandwidth = getTheoreticalPeakBandwidth();
            printf("Approximate theoretical peak FP32: %.2f GFLOPS\n", peakGflops);
            printf("Approximate theoretical peak bandwidth: %.2f GB/s\n", peakBandwidth);
        } else {
            printf("Approximate theoretical peak FP32: unavailable without a CUDA device\n");
            printf("Approximate theoretical peak bandwidth: unavailable without a CUDA device\n");
        }
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
        bool has_tensor_cores = tensorCoresAvailable();

        // cuBLAS 参考
        printf("\nRunning cuBLAS (reference)...\n");
        BenchmarkResult cublas_result = benchmark.runCublas(
            M, K, N, config_.settings.run.warmup_runs, config_.settings.run.benchmark_runs);
        float cublas_gflops = cublas_result.gflops;

        // 使用 Catalog 驱动的内核调度
        runCatalogKernels(benchmark, M, K, N, has_tensor_cores);

        // Tensor Core compute-only 是特殊情况（需要 cublas handle）
        runTensorCoreComputeOnly(benchmark, M, K, N, has_tensor_cores);

        // 报告
        benchmark.printSummary();
        printPerformanceComparison(benchmark.getResults(), cublas_gflops);

        // 导出 roofline 数据 (仅当设置启用时)
        if (config_.settings.output.export_roofline) {
            std::string filename = config_.settings.output.makeRooflineFilename(M, K, N);
            benchmark.exportRooflineData(filename);
        }
    }

    void runCatalogKernels(SGEMMBenchmark &benchmark, int M, int K, int N, bool has_tensor_cores) {
        const auto &catalog = getKernelCatalog();

        for (const auto &entry : catalog) {
            // Check runtime constraints from catalog
            if (!entry.canRun(M, K, N, has_tensor_cores)) {
                printSkipReason(entry, M, K, N, has_tensor_cores);
                continue;
            }

            // Use tolerance from settings (which may override defaults)
            VerifyTolerance tolerance = config_.settings.toleranceForKernel(entry.type);

            printf("Running %s SGEMM...\n", formatConsoleName(entry.name).c_str());
            benchmark.run(entry.name, entry.launcher, M, K, N, config_.settings.run.warmup_runs,
                          config_.settings.run.benchmark_runs, tolerance);
        }
    }

    void runTensorCoreComputeOnly(SGEMMBenchmark &benchmark, int M, int K, int N,
                                  bool has_tensor_cores) {
        auto tc_entry = getTensorCoreComputeOnlyEntry();

        if (!tc_entry.canRun(M, K, N, has_tensor_cores)) {
            return; // Skip silently - already reported by catalog kernels
        }

        VerifyTolerance tolerance = config_.settings.toleranceForKernel(KernelType::TensorCore);
        printf("Running Tensor Core SGEMM (compute-only WMMA path)...\n");

        BenchmarkResult tc_result = runTensorCoreComputeOnlyBenchmark(
            benchmark.getCublasHandle(), M, K, N, config_.settings.run.warmup_runs,
            config_.settings.run.benchmark_runs, tolerance);
        benchmark.addResult(tc_result);
    }

    void printSkipReason(const KernelCatalogEntry &entry, int M, int K, int N,
                         bool has_tensor_cores) const {
        if (entry.constraints.requires_tensor_cores && !has_tensor_cores) {
            // Use non-throwing error handling for diagnostic output
            int device = 0;
            cudaDeviceProp prop{};
            if (cudaGetDevice(&device) == cudaSuccess &&
                cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
                printf("Skipping %s (requires sm_70+, current: sm_%d%d)\n", entry.name.c_str(),
                       prop.major, prop.minor);
            } else {
                printf("Skipping %s (requires sm_70+, current device unavailable)\n",
                       entry.name.c_str());
            }
        } else if (entry.constraints.dimension_alignment > 0) {
            printf("Skipping %s for %d x %d x %d "
                   "(requires dimensions aligned to %d)\n",
                   entry.name.c_str(), M, K, N, entry.constraints.dimension_alignment);
        }
    }

    std::string formatConsoleName(const std::string &name) const {
        // Strip tile size annotation for console message
        std::string result = name;
        size_t paren_pos = result.find(" (");
        if (paren_pos != std::string::npos) {
            result = result.substr(0, paren_pos);
        }
        return result;
    }

    BenchmarkConfig config_;
};
