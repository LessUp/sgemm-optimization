/**
 * SGEMM Optimization Benchmark
 *
 * This program benchmarks various SGEMM implementations from naive to
 * Tensor Core optimized, demonstrating the performance evolution of
 * GPU matrix multiplication optimization techniques.
 */

#include <cstdio>
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>

#include "utils/benchmark.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/verify.cuh"

#include "kernels/bank_conflict_free_sgemm.cuh"
#include "kernels/double_buffer_sgemm.cuh"
#include "kernels/naive_sgemm.cuh"
#include "kernels/tensor_core_sgemm.cuh"
#include "kernels/tiled_sgemm.cuh"

namespace {
int warmup_runs = 5;
int benchmark_runs = 20;

const std::vector<std::tuple<int, int, int>> DEFAULT_CASES = {
    {512, 512, 512},
    {1024, 1024, 1024},
    {256, 384, 640},
    {511, 513, 1025},
};
}

void naive_kernel(const float *A, const float *B, float *C, int M, int K,
                  int N) {
  launch_naive_sgemm<32>(A, B, C, M, K, N);
}

void tiled_kernel(const float *A, const float *B, float *C, int M, int K,
                  int N) {
  launch_tiled_sgemm<32>(A, B, C, M, K, N);
}

void bank_conflict_free_kernel(const float *A, const float *B, float *C, int M,
                               int K, int N) {
  launch_bank_conflict_free_sgemm<32>(A, B, C, M, K, N);
}

void double_buffer_kernel(const float *A, const float *B, float *C, int M,
                          int K, int N) {
  launch_double_buffer_sgemm<32>(A, B, C, M, K, N);
}

void tensor_core_kernel(const float *A, const float *B, float *C, int M, int K,
                        int N) {
  launch_tensor_core_sgemm(A, B, C, M, K, N);
}

void runBenchmarks(int M, int K, int N) {
  printf("\n");
  printf("====================================================================="
         "===========\n");
  printf("                    Benchmarking %d x %d x %d SGEMM\n", M, K, N);
  printf("====================================================================="
         "===========\n");

  SGEMMBenchmark benchmark;

  printf("\nRunning cuBLAS (reference)...\n");
  BenchmarkResult cublas_result =
      benchmark.runCublas(M, K, N, warmup_runs, benchmark_runs);
  float cublas_gflops = cublas_result.gflops;

  printf("Running Naive SGEMM...\n");
  benchmark.run("Naive", naive_kernel, M, K, N, warmup_runs, benchmark_runs,
                kStandardVerifyTolerance);

  printf("Running Tiled SGEMM...\n");
  benchmark.run("Tiled (32x32)", tiled_kernel, M, K, N, warmup_runs,
                benchmark_runs, kStandardVerifyTolerance);

  printf("Running Bank Conflict Free SGEMM...\n");
  benchmark.run("Bank Conflict Free", bank_conflict_free_kernel, M, K, N,
                warmup_runs, benchmark_runs, kStandardVerifyTolerance);

  printf("Running Double Buffer SGEMM...\n");
  benchmark.run("Double Buffer", double_buffer_kernel, M, K, N, warmup_runs,
                benchmark_runs, kStandardVerifyTolerance);

  if (tensorCoresAvailable()) {
    printf("Running Tensor Core SGEMM (end-to-end, includes FP32->FP16 "
           "conversion/fallback)...\n");
    benchmark.run("Tensor Core (WMMA end-to-end)", tensor_core_kernel, M, K, N,
                  warmup_runs, benchmark_runs, kTensorCoreVerifyTolerance);

    if (tensorCoreDimensionsSupported(M, K, N)) {
      printf("Running Tensor Core SGEMM (compute-only WMMA path)...\n");
      benchmark.runTensorCoreComputeOnly(M, K, N, warmup_runs,
                                         benchmark_runs,
                                         kTensorCoreVerifyTolerance);
    } else {
      printf("Skipping Tensor Core compute-only benchmark (requires positive "
             "dimensions aligned to 16).\n");
    }
  } else {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Skipping Tensor Core benchmarks (requires sm_70+, current: sm_%d%d)\n",
           prop.major, prop.minor);
  }

  benchmark.printSummary();
  printPerformanceComparison(benchmark.getResults(), cublas_gflops);

  char filename[256];
  snprintf(filename, sizeof(filename), "roofline_data_%d_%d_%d.csv", M, K, N);
  benchmark.exportRooflineData(filename);
}

void printUsage(const char *program) {
  printf("Usage: %s [options]\n", program);
  printf("\nOptions:\n");
  printf("  -s, --size SIZE          Benchmark one square SIZE x SIZE x SIZE case\n");
  printf("  --dims M K N            Benchmark one explicit M x K x N case\n");
  printf("  -a, --all               Run the default benchmark set\n");
  printf("  --warmup N              Number of warmup runs (default: 5)\n");
  printf("  --benchmark N           Number of benchmark runs (default: 20)\n");
  printf("  -h, --help              Show this help message\n");
  printf("\nDefault benchmark set includes:\n");
  printf("  - aligned square cases (512, 1024)\n");
  printf("  - one aligned non-square case (256 x 384 x 640)\n");
  printf("  - one unaligned edge case (511 x 513 x 1025) to exercise safe Tensor Core fallback\n");
  printf("\nExamples:\n");
  printf("  %s -s 1024\n", program);
  printf("  %s --dims 256 384 640\n", program);
  printf("  %s -a --warmup 10 --benchmark 50\n", program);
}

int main(int argc, char **argv) {
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

  std::vector<std::tuple<int, int, int>> cases;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      printUsage(argv[0]);
      return 0;
    }

    if (arg == "-s" || arg == "--size") {
      if (i + 1 >= argc) {
        fprintf(stderr, "Error: -s requires a size argument\n");
        return 1;
      }

      int size = atoi(argv[++i]);
      if (size <= 0) {
        fprintf(stderr, "Error: Size must be positive\n");
        return 1;
      }

      cases.emplace_back(size, size, size);
      continue;
    }

    if (arg == "--dims") {
      if (i + 3 >= argc) {
        fprintf(stderr, "Error: --dims requires M K N arguments\n");
        return 1;
      }

      int M = atoi(argv[++i]);
      int K = atoi(argv[++i]);
      int N = atoi(argv[++i]);
      if (M <= 0 || K <= 0 || N <= 0) {
        fprintf(stderr, "Error: Dimensions must be positive\n");
        return 1;
      }

      cases.emplace_back(M, K, N);
      continue;
    }

    if (arg == "-a" || arg == "--all") {
      cases = DEFAULT_CASES;
      continue;
    }

    if (arg == "--warmup") {
      if (i + 1 >= argc) {
        fprintf(stderr, "Error: --warmup requires a number argument\n");
        return 1;
      }

      int warmup = atoi(argv[++i]);
      if (warmup < 0) {
        fprintf(stderr, "Error: Warmup runs must be non-negative\n");
        return 1;
      }

      warmup_runs = warmup;
      continue;
    }

    if (arg == "--benchmark") {
      if (i + 1 >= argc) {
        fprintf(stderr, "Error: --benchmark requires a number argument\n");
        return 1;
      }

      int bench = atoi(argv[++i]);
      if (bench <= 0) {
        fprintf(stderr, "Error: Benchmark runs must be positive\n");
        return 1;
      }

      benchmark_runs = bench;
      continue;
    }

    fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
    printUsage(argv[0]);
    return 1;
  }

  if (cases.empty()) {
    cases.emplace_back(1024, 1024, 1024);
  }

  for (const auto &[M, K, N] : cases) {
    runBenchmarks(M, K, N);
  }

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
  printf("  - The end-to-end Tensor Core result includes FP32->FP16 conversion and safe fallback behavior.\n");
  printf("  - The compute-only Tensor Core result is only shown for WMMA-compatible dimensions.\n");
  printf("\n");

  return 0;
}
