#pragma once

#include "../kernels/tensor_core_sgemm.cuh"
#include "cuda_utils.cuh"
#include "verify.cuh"
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <string>
#include <vector>

// ============================================================================
// Benchmark Result Structure
// ============================================================================

struct BenchmarkResult {
  std::string kernel_name;
  int M, K, N;
  float time_ms;
  float gflops;
  float bandwidth_gb_s;
  bool correct;
  float max_error;
  float efficiency; // Percentage of theoretical peak

  void print() const {
    printf("  %-30s | %4d x %4d x %4d | %8.3f ms | %8.2f GFLOPS | %s | err: "
           "%.2e\n",
           kernel_name.c_str(), M, K, N, time_ms, gflops,
           correct ? "PASS" : "FAIL", max_error);
  }
};

// ============================================================================
// SGEMM Benchmark Class
// ============================================================================

class SGEMMBenchmark {
public:
  SGEMMBenchmark() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  }

  ~SGEMMBenchmark() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
    cublasDestroy(cublas_handle_);
  }

  template <typename KernelFunc>
  BenchmarkResult run(const std::string &name, KernelFunc kernel_func, int M,
                      int K, int N, int warmup_runs = 5,
                      int benchmark_runs = 20,
                      VerifyTolerance tolerance = kStandardVerifyTolerance) {
    BenchmarkResult result;
    result.kernel_name = name;
    result.M = M;
    result.K = K;
    result.N = N;

    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_C_ref(M * N);
    DeviceMemory<float> d_A(M * K);
    DeviceMemory<float> d_B(K * N);
    DeviceMemory<float> d_C(M * N);
    DeviceMemory<float> d_C_ref(M * N);

    initRandomMatrix(h_A.data(), M, K, -1.0f, 1.0f, 42);
    initRandomMatrix(h_B.data(), K, N, -1.0f, 1.0f, 123);

    d_A.copyFromHost(h_A.data(), M * K);
    d_B.copyFromHost(h_B.data(), K * N);

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &alpha, d_B.get(), N, d_A.get(), K, &beta,
                             d_C_ref.get(), N));

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
    fillPerformanceMetrics(result, total_time_ms / benchmark_runs);

    d_C.copyToHost(h_C.data(), M * N);
    d_C_ref.copyToHost(h_C_ref.data(), M * N);

    VerifyResult verify_result = compareMatrices(h_C.data(), h_C_ref.data(), M,
                                                 N, tolerance);
    result.correct = verify_result.passed;
    result.max_error = verify_result.max_rel_error;

    results_.push_back(result);
    return result;
  }

  BenchmarkResult runCublas(int M, int K, int N, int warmup_runs = 5,
                            int benchmark_runs = 20) {
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
      CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M,
                               K, &alpha, d_B.get(), N, d_A.get(), K, &beta,
                               d_C.get(), N));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start_));
    for (int i = 0; i < benchmark_runs; ++i) {
      CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M,
                               K, &alpha, d_B.get(), N, d_A.get(), K, &beta,
                               d_C.get(), N));
    }
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));

    float total_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_time_ms, start_, stop_));
    fillPerformanceMetrics(result, total_time_ms / benchmark_runs);

    result.correct = true;
    result.max_error = 0.0f;
    results_.push_back(result);
    return result;
  }

  BenchmarkResult runTensorCoreComputeOnly(
      int M, int K, int N, int warmup_runs = 5, int benchmark_runs = 20,
      VerifyTolerance tolerance = kTensorCoreVerifyTolerance) {
    if (!tensorCoresAvailable() || !tensorCoreDimensionsSupported(M, K, N)) {
      throw CudaError("Tensor Core compute-only benchmark requires sm_70+ and "
                      "dimensions aligned to 16");
    }

    BenchmarkResult result;
    result.kernel_name = "Tensor Core (WMMA compute-only)";
    result.M = M;
    result.K = K;
    result.N = N;

    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_C_ref(M * N);
    DeviceMemory<float> d_A(M * K);
    DeviceMemory<float> d_B(K * N);
    DeviceMemory<float> d_C(M * N);
    DeviceMemory<float> d_C_ref(M * N);
    DeviceMemory<half> d_A_fp16(M * K);
    DeviceMemory<half> d_B_fp16(K * N);

    initRandomMatrix(h_A.data(), M, K, -1.0f, 1.0f, 42);
    initRandomMatrix(h_B.data(), K, N, -1.0f, 1.0f, 123);

    d_A.copyFromHost(h_A.data(), M * K);
    d_B.copyFromHost(h_B.data(), K * N);

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &alpha, d_B.get(), N, d_A.get(), K, &beta,
                             d_C_ref.get(), N));

    int blockSize = 256;
    int gridSizeA = (M * K + blockSize - 1) / blockSize;
    int gridSizeB = (K * N + blockSize - 1) / blockSize;

    float_to_half_kernel<<<gridSizeA, blockSize>>>(d_A.get(), d_A_fp16.get(),
                                                   M * K);
    float_to_half_kernel<<<gridSizeB, blockSize>>>(d_B.get(), d_B_fp16.get(),
                                                   K * N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < warmup_runs; ++i) {
      d_C.zero();
      launch_tensor_core_sgemm_fp16(d_A_fp16.get(), d_B_fp16.get(), d_C.get(), M,
                                    K, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start_));
    for (int i = 0; i < benchmark_runs; ++i) {
      launch_tensor_core_sgemm_fp16(d_A_fp16.get(), d_B_fp16.get(), d_C.get(), M,
                                    K, N);
    }
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));

    float total_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_time_ms, start_, stop_));
    fillPerformanceMetrics(result, total_time_ms / benchmark_runs);

    d_C.copyToHost(h_C.data(), M * N);
    d_C_ref.copyToHost(h_C_ref.data(), M * N);

    VerifyResult verify_result = compareMatrices(h_C.data(), h_C_ref.data(), M,
                                                 N, tolerance);
    result.correct = verify_result.passed;
    result.max_error = verify_result.max_rel_error;

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
    printf("  %-30s | %-17s | %10s | %14s | %4s | %s\n", "Kernel",
           "Dimensions", "Time", "Performance", "Pass", "Max Error");
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
          (result.M * result.K + result.K * result.N + result.M * result.N) *
          sizeof(float);
      double ai = flops / bytes;

      file << result.kernel_name << "," << result.M << "," << result.K << ","
           << result.N << "," << result.time_ms << "," << result.gflops << ","
           << result.bandwidth_gb_s << "," << ai << "\n";
    }

    file.close();
    printf("Approximate roofline data exported to: %s\n", filename.c_str());
  }

  const std::vector<BenchmarkResult> &getResults() const { return results_; }

  void clearResults() { results_.clear(); }

  cublasHandle_t getCublasHandle() const { return cublas_handle_; }

private:
  static void fillPerformanceMetrics(BenchmarkResult &result, float time_ms) {
    result.time_ms = time_ms;
    double flops = 2.0 * result.M * result.N * result.K;
    result.gflops = (flops / (result.time_ms * 1e-3)) / 1e9;

    double bytes = (result.M * result.K + result.K * result.N +
                    result.M * result.N) *
                   sizeof(float);
    result.bandwidth_gb_s = (bytes / (result.time_ms * 1e-3)) / 1e9;
  }

  cudaEvent_t start_, stop_;
  cublasHandle_t cublas_handle_;
  std::vector<BenchmarkResult> results_;
};

// ============================================================================
// Utility Functions
// ============================================================================

inline void
printPerformanceComparison(const std::vector<BenchmarkResult> &results,
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
    printf("  %-30s | %10.2f     | %8.1f%%\n", result.kernel_name.c_str(),
           result.gflops, percentage);
  }
  printf("---------------------------------------------------------------------"
         "-----------\n");
}

inline float getTheoreticalPeakGflops() {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  // Cores per SM based on architecture
  // See: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
  int coresPerSM;
  if (prop.major == 7) {
    coresPerSM = (prop.minor == 0 || prop.minor == 2) ? 64 : 64; // Volta, Turing
  } else if (prop.major == 8) {
    coresPerSM = (prop.minor == 0 || prop.minor == 6) ? 64 : 128; // Ampere
  } else if (prop.major == 9) {
    coresPerSM = 128; // Hopper
  } else {
    coresPerSM = 64; // Default fallback
  }

  // Use actual GPU clock rate (convert from kHz to GHz)
  float clockGHz = static_cast<float>(prop.clockRate) / 1e6f;

  // Peak GFLOPS = SMs * cores/SM * 2 (FMA) * clock (GHz) * 1000 (MHz factor)
  float peakGflops =
      prop.multiProcessorCount * coresPerSM * 2 * clockGHz * 1000;

  return peakGflops;
}

inline float getTheoreticalPeakBandwidth() {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  // Calculate peak bandwidth using memory clock rate and bus width
  // Note: memoryClockRate may be 0 on some platforms/drivers
  float memoryClockMHz = static_cast<float>(prop.memoryClockRate) / 1000.0f;

  // If memoryClockRate is not available, use a reasonable default based on GPU generation
  if (memoryClockMHz <= 0) {
    // Fallback: estimate based on typical values for the architecture
    // This is a rough approximation; actual values vary significantly
    switch (prop.major) {
    case 7:
      memoryClockMHz = (prop.minor == 5) ? 1750.0f : 877.0f; // Turing vs Volta
      break;
    case 8:
      memoryClockMHz = (prop.minor == 6) ? 1215.0f : 1593.0f; // A100 vs RTX 30
      break;
    case 9:
      memoryClockMHz = 2619.0f; // H100 HBM3
      break;
    default:
      memoryClockMHz = 1000.0f; // Conservative default
    }
  }

  // Peak bandwidth = 2 (DDR) * clock (MHz) * bus width (bits) / 8 (bytes)
  float peakBandwidth = 2 * memoryClockMHz * (prop.memoryBusWidth / 8) / 1000.0f;

  return peakBandwidth; // GB/s
}
