#pragma once

#include "../utils/benchmark.cuh"
#include "tensor_core_capabilities.cuh"
#include "tensor_core_compute.cuh"
#include "tensor_core_launcher.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

// ============================================================================
// Tensor Core Benchmark Extensions
// ============================================================================
//
// 此模块提供 Tensor Core 特有的 benchmark 功能，将工具层的
// Tensor Core 特定逻辑移到内核层，避免循环依赖。
//
// 架构原理：
// - benchmark.cuh (工具层) 只提供通用的 benchmark 框架
// - tensor_core_benchmark.cuh (内核层) 提供 Tensor Core 特有的 benchmark 功能
// ============================================================================

/**
 * 运行 Tensor Core 纯计算路径 benchmark
 *
 * 此函数仅测试 WMMA FP16→FP32 计算性能，不包含 FP32→FP16 转换和 fallback。
 * 用于分离测量 Tensor Core 计算单元的实际性能。
 *
 * @param benchmark SGEMMBenchmark 实例
 * @param M, K, N 矩阵维度
 * @param warmup_runs 预热次数
 * @param benchmark_runs 测量次数
 * @param tolerance 验证容差
 * @return BenchmarkResult 包含性能数据
 */
inline BenchmarkResult
runTensorCoreComputeOnlyBenchmark(SGEMMBenchmark &benchmark, int M, int K, int N,
                                  int warmup_runs = 5, int benchmark_runs = 20,
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
    CUBLAS_CHECK(cublasSgemm(benchmark.getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             d_B.get(), N, d_A.get(), K, &beta, d_C_ref.get(), N));

    int blockSize = 256;
    int gridSizeA = (M * K + blockSize - 1) / blockSize;
    int gridSizeB = (K * N + blockSize - 1) / blockSize;

    float_to_half_kernel<<<gridSizeA, blockSize>>>(d_A.get(), d_A_fp16.get(), M * K);
    float_to_half_kernel<<<gridSizeB, blockSize>>>(d_B.get(), d_B_fp16.get(), K * N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < warmup_runs; ++i) {
        d_C.zero();
        launch_tensor_core_sgemm_fp16(d_A_fp16.get(), d_B_fp16.get(), d_C.get(), M, K, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < benchmark_runs; ++i) {
        launch_tensor_core_sgemm_fp16(d_A_fp16.get(), d_B_fp16.get(), d_C.get(), M, K, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_time_ms, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 填充性能指标
    result.time_ms = total_time_ms / benchmark_runs;
    double flops = 2.0 * result.M * result.N * result.K;
    result.gflops = (flops / (result.time_ms * 1e-3)) / 1e9;
    double bytes =
        (result.M * result.K + result.K * result.N + result.M * result.N) * sizeof(float);
    result.bandwidth_gb_s = (bytes / (result.time_ms * 1e-3)) / 1e9;

    d_C.copyToHost(h_C.data(), M * N);
    d_C_ref.copyToHost(h_C_ref.data(), M * N);

    VerifyResult verify_result = compareMatrices(h_C.data(), h_C_ref.data(), M, N, tolerance);
    result.correct = verify_result.passed;
    result.max_error = verify_result.max_rel_error;

    return result;
}
