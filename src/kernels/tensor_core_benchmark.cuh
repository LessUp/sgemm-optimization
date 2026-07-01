#pragma once

#include "../utils/benchmark_core.cuh"
#include "../utils/benchmark_metrics.cuh"
#include "../utils/verify.cuh"
#include "tensor_core_sgemm.cuh"

#include <climits>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

// ============================================================================
// Tensor Core Benchmark Extensions
// ============================================================================
//
// 此模块提供 Tensor Core 特有的 benchmark 功能。
// 接口设计：只接受 cublasHandle_t，不依赖整个 SGEMMBenchmark 类，
// 避免内核层对工具层的上穿依赖。
//
// 设计原则：
// - 与 KernelCatalog 的约束保持一致
// - 使用统一的容差和验证策略
// - 所有约束检查集中在 tensorCoreDimensionsSupported() 和 tensorCoresAvailable()
// ============================================================================

/**
 * 检查 Tensor Core compute-only benchmark 是否可以运行
 *
 * 统一的约束检查，与 KernelCatalog 的 canRun() 语义一致
 */
inline bool canRunTensorCoreComputeOnly(int M, int K, int N) {
    return tensorCoresAvailable() && tensorCoreDimensionsSupported(M, K, N);
}

/**
 * 运行 Tensor Core 纯计算路径 benchmark
 *
 * 此函数仅测试 WMMA FP16→FP32 计算性能，不包含 FP32→FP16 转换和 fallback。
 * 用于分离测量 Tensor Core 计算单元的实际性能。
 *
 * @param cublas_handle cuBLAS 句柄（用于参考计算）
 * @param M, K, N 矩阵维度
 * @param warmup_runs 预热次数
 * @param benchmark_runs 测量次数
 * @param tolerance 验证容差
 * @return BenchmarkResult 包含性能数据
 * @throws CudaError 如果约束不满足
 */
inline BenchmarkResult
runTensorCoreComputeOnlyBenchmark(cublasHandle_t cublas_handle, int M, int K, int N,
                                  int warmup_runs = 5, int benchmark_runs = 20,
                                  VerifyTolerance tolerance = kTensorCoreVerifyTolerance) {

    // 约束检查 - 与 KernelCatalog 语义一致
    if (!canRunTensorCoreComputeOnly(M, K, N)) {
        throw CudaError("Tensor Core compute-only benchmark requires sm_70+ and "
                        "dimensions aligned to 16");
    }

    BenchmarkResult result;
    result.kernel_name = "Tensor Core (WMMA compute-only)";
    result.M = M;
    result.K = K;
    result.N = N;

    // 安全计算矩阵大小，避免整数溢出
    size_t size_A = static_cast<size_t>(M) * K;
    size_t size_B = static_cast<size_t>(K) * N;
    size_t size_C = static_cast<size_t>(M) * N;

    std::vector<float> h_A(size_A), h_B(size_B), h_C(size_C), h_C_ref(size_C);
    DeviceMemory<float> d_A(size_A);
    DeviceMemory<float> d_B(size_B);
    DeviceMemory<float> d_C(size_C);
    DeviceMemory<float> d_C_ref(size_C);
    DeviceMemory<half> d_A_fp16(size_A);
    DeviceMemory<half> d_B_fp16(size_B);

    initRandomMatrix(h_A.data(), M, K, -1.0f, 1.0f, 42);
    initRandomMatrix(h_B.data(), K, N, -1.0f, 1.0f, 123);

    d_A.copyFromHost(h_A.data(), size_A);
    d_B.copyFromHost(h_B.data(), size_B);

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B.get(), N,
                             d_A.get(), K, &beta, d_C_ref.get(), N));

    int blockSize = kDefaultBlockSize;
    checkMatrixElementCount(size_A, "A");
    checkMatrixElementCount(size_B, "B");
    int gridSizeA = safeGridSize(size_A, blockSize);
    int gridSizeB = safeGridSize(size_B, blockSize);

    float_to_half_kernel<<<gridSizeA, blockSize>>>(d_A.get(), d_A_fp16.get(),
                                                   static_cast<int>(size_A));
    float_to_half_kernel<<<gridSizeB, blockSize>>>(d_B.get(), d_B_fp16.get(),
                                                   static_cast<int>(size_B));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 使用统一的 measureGpuTime 计时
    float time_ms = measureGpuTime(
        [&]() {
            launch_tensor_core_sgemm_fp16(d_A_fp16.get(), d_B_fp16.get(), d_C.get(), M, K, N);
        },
        warmup_runs, benchmark_runs);

    // 计算指标
    PerformanceMetrics metrics = calculateSgemmMetrics(M, K, N, time_ms);
    result.time_ms = metrics.time_ms;
    result.gflops = metrics.gflops;
    result.bandwidth_gb_s = metrics.bandwidth_gb_s;
    result.efficiency = calculateEfficiency(result.gflops, getTheoreticalPeakGflops());

    d_C.copyToHost(h_C.data(), size_C);
    d_C_ref.copyToHost(h_C_ref.data(), size_C);

    VerifyResult verify_result = compareMatrices(h_C.data(), h_C_ref.data(), M, N, tolerance);
    result.correct = verify_result.passed;
    result.max_error = verify_result.max_rel_error;

    return result;
}
