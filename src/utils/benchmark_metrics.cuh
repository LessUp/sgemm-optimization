#pragma once

#include "cuda_utils.cuh"
#include <cstdio>
#include <cuda_runtime.h>

// ============================================================================
// 性能指标结构
// ============================================================================

/**
 * 存储计算的性能指标
 */
struct PerformanceMetrics {
    float time_ms;           // 执行时间（毫秒）
    float gflops;            // GFLOPS（每秒十亿次浮点运算）
    float bandwidth_gb_s;    // 带宽（GB/s）
    float arithmetic_intensity; // 算术强度（FLOPs/Byte）
};

// ============================================================================
// 指标计算函数
// ============================================================================

/**
 * 计算 SGEMM 的性能指标
 *
 * @param M, K, N 矩阵维度
 * @param time_ms 执行时间（毫秒）
 * @return 性能指标
 */
inline PerformanceMetrics calculateSgemmMetrics(int M, int K, int N, float time_ms) {
    PerformanceMetrics metrics;
    metrics.time_ms = time_ms;

    // SGEMM: C = A * B
    // 浮点运算次数: 2 * M * N * K (每次乘加为 2 次运算)
    double flops = 2.0 * M * N * K;
    metrics.gflops = (flops / (time_ms * 1e-3)) / 1e9;

    // 数据传输量（字节）: A(M*K) + B(K*N) + C(M*N)
    double bytes = (M * K + K * N + M * N) * sizeof(float);
    metrics.bandwidth_gb_s = (bytes / (time_ms * 1e-3)) / 1e9;

    // 算术强度: FLOPs / Bytes
    metrics.arithmetic_intensity = flops / bytes;

    return metrics;
}

/**
 * 获取 GPU 理论峰值 GFLOPS
 *
 * 基于 GPU 架构和时钟频率计算：
 * - SM 数量
 * - 每 SM 的 CUDA 核心数
 * - 时钟频率
 */
inline float getTheoreticalPeakGflops() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // 每个 SM 的核心数（基于架构）
    // 参考: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
    int coresPerSM;
    if (prop.major == 7) {
        coresPerSM = 64; // Volta (sm_70, sm_72), Turing (sm_75)
    } else if (prop.major == 8) {
        coresPerSM = (prop.minor == 0 || prop.minor == 6)
                         ? 64
                         : 128; // A100/sm_80, A10G/sm_86: 64, others: 128
    } else if (prop.major == 9) {
        coresPerSM = 128; // Hopper (sm_90)
    } else {
        coresPerSM = 64; // 默认回退
    }

    // 时钟频率 (kHz -> GHz)
    float clockGHz = static_cast<float>(prop.clockRate) / 1e6f;

    // 峰值 GFLOPS = SMs * cores/SM * 2 (FMA) * clock (GHz) * 1000 (MHz factor)
    float peakGflops = prop.multiProcessorCount * coresPerSM * 2 * clockGHz * 1000;

    return peakGflops;
}

/**
 * 获取 GPU 理论峰值带宽 (GB/s)
 *
 * 基于内存时钟频率和总线宽度计算：
 * - DDR 倍率 (2x)
 * - 内存时钟频率
 * - 内存总线宽度
 */
inline float getTheoreticalPeakBandwidth() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // 内存时钟频率 (Hz -> MHz)
    float memoryClockMHz = static_cast<float>(prop.memoryClockRate) / 1000.0f;

    // 如果 memoryClockRate 不可用，使用架构默认值
    if (memoryClockMHz <= 0) {
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
            memoryClockMHz = 1000.0f; // 保守默认值
        }
    }

    // 峰值带宽 = 2 (DDR) * clock (MHz) * bus width (bits) / 8 (bytes)
    float peakBandwidth = 2 * memoryClockMHz * (prop.memoryBusWidth / 8) / 1000.0f;

    return peakBandwidth; // GB/s
}

/**
 * 计算效率（相对于理论峰值的百分比）
 */
inline float calculateEfficiency(float actual_gflops, float peak_gflops) {
    if (peak_gflops <= 0)
        return 0.0f;
    return (actual_gflops / peak_gflops) * 100.0f;
}

/**
 * 计算带宽利用率（相对于理论峰值的百分比）
 */
inline float calculateBandwidthUtilization(float actual_bandwidth, float peak_bandwidth) {
    if (peak_bandwidth <= 0)
        return 0.0f;
    return (actual_bandwidth / peak_bandwidth) * 100.0f;
}

// ============================================================================
// 性能比较工具
// ============================================================================

/**
 * 打印性能比较报告
 *
 * @param kernel_name 内核名称
 * @param metrics 性能指标
 * @param baseline_gflops 基线 GFLOPS（如 cuBLAS）
 */
inline void printPerformanceReport(const char* kernel_name, const PerformanceMetrics& metrics,
                                   float baseline_gflops = 0.0f) {
    printf("  %-30s | %8.3f ms | %10.2f GFLOPS | %8.2f GB/s | AI: %.1f\n", kernel_name,
           metrics.time_ms, metrics.gflops, metrics.bandwidth_gb_s, metrics.arithmetic_intensity);

    if (baseline_gflops > 0.0f) {
        float percentage = (metrics.gflops / baseline_gflops) * 100.0f;
        printf("    -> %.1f%% of baseline (%.2f GFLOPS)\n", percentage, baseline_gflops);
    }
}
