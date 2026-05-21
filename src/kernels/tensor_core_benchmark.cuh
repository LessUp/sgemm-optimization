#pragma once

#include "tensor_core_sgemm.cuh"
#include "../utils/benchmark_core.cuh"
#include "../utils/benchmark_metrics.cuh"
#include "../utils/verify.cuh"

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
// ============================================================================

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
 */
inline BenchmarkResult
runTensorCoreComputeOnlyBenchmark(cublasHandle_t cublas_handle, int M, int K, int N,
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
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             d_B.get(), N, d_A.get(), K, &beta, d_C_ref.get(), N));

    int blockSize = kDefaultBlockSize;
    // 安全计算 gridSize，检查溢出
    auto safeGridSize = [](size_t num, int blk) -> int {
        size_t grid = (num + blk - 1) / blk;
        if (grid > static_cast<size_t>(INT_MAX)) {
            throw CudaError("Grid size overflow: matrix too large for kernel launch");
        }
        return static_cast<int>(grid);
    };
    int gridSizeA = safeGridSize(static_cast<size_t>(M) * K, blockSize);
    int gridSizeB = safeGridSize(static_cast<size_t>(K) * N, blockSize);

    size_t num_A = static_cast<size_t>(M) * K;
    size_t num_B = static_cast<size_t>(K) * N;

    // 检查矩阵元素数量是否超过 int 最大值
    if (num_A > static_cast<size_t>(INT_MAX)) {
        throw CudaError("Matrix A size overflow: too many elements for int parameter");
    }
    if (num_B > static_cast<size_t>(INT_MAX)) {
        throw CudaError("Matrix B size overflow: too many elements for int parameter");
    }

    float_to_half_kernel<<<gridSizeA, blockSize>>>(d_A.get(), d_A_fp16.get(), static_cast<int>(num_A));
    float_to_half_kernel<<<gridSizeB, blockSize>>>(d_B.get(), d_B_fp16.get(), static_cast<int>(num_B));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 使用统一的 measureGpuTime 计时
    float time_ms =
        measureGpuTime([&]() { launch_tensor_core_sgemm_fp16(d_A_fp16.get(), d_B_fp16.get(), d_C.get(), M, K, N); },
                       warmup_runs, benchmark_runs);

    // 计算指标
    PerformanceMetrics metrics = calculateSgemmMetrics(M, K, N, time_ms);
    result.time_ms = metrics.time_ms;
    result.gflops = metrics.gflops;
    result.bandwidth_gb_s = metrics.bandwidth_gb_s;
    result.efficiency = calculateEfficiency(result.gflops, getTheoreticalPeakGflops());

    d_C.copyToHost(h_C.data(), M * N);
    d_C_ref.copyToHost(h_C_ref.data(), M * N);

    VerifyResult verify_result = compareMatrices(h_C.data(), h_C_ref.data(), M, N, tolerance);
    result.correct = verify_result.passed;
    result.max_error = verify_result.max_rel_error;

    return result;
}
