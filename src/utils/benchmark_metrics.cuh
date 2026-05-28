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
    float time_ms;              // 执行时间（毫秒）
    float gflops;               // GFLOPS（每秒十亿次浮点运算）
    float bandwidth_gb_s;       // 带宽（GB/s）
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
 *
 * 提供重载版本以支持可注入的 device info provider。
 */
inline float getTheoreticalPeakGflops(const DeviceInfoProvider &provider) {
    // 峰值 GFLOPS = SMs * cores/SM * 2 (FMA) * clock (GHz) * 1000 (MHz factor)
    float peakGflops = provider.smCount() * provider.cores_per_sm * 2 * provider.clock_ghz * 1000;
    return peakGflops;
}

/**
 * 获取 GPU 理论峰值 GFLOPS（默认使用生产环境设备）
 */
inline float getTheoreticalPeakGflops() {
    if (!cudaDeviceAvailable()) {
        return 0.0f;
    }
    return getTheoreticalPeakGflops(getProductionDeviceInfo());
}

/**
 * 获取 GPU 理论峰值带宽 (GB/s)
 *
 * 基于内存时钟频率和总线宽度计算：
 * - DDR 倍率 (2x)
 * - 内存时钟频率
 * - 内存总线宽度
 *
 * 提供重载版本以支持可注入的 device info provider。
 */
inline float getTheoreticalPeakBandwidth(const DeviceInfoProvider &provider) {
    // 内存时钟频率 (Hz -> MHz)
    float memoryClockMHz = static_cast<float>(provider.memoryClockRate()) / 1000.0f;

    // 如果 memoryClockRate 不可用，使用架构默认值
    if (memoryClockMHz <= 0) {
        int major = provider.computeMajor();
        int minor = provider.computeMinor();
        switch (major) {
        case 7:
            memoryClockMHz = (minor == 5) ? 1750.0f : 877.0f; // Turing vs Volta
            break;
        case 8:
            memoryClockMHz = (minor == 6) ? 1215.0f : 1593.0f; // A100 vs RTX 30
            break;
        case 9:
            memoryClockMHz = 2619.0f; // H100 HBM3
            break;
        default:
            memoryClockMHz = 1000.0f; // 保守默认值
        }
    }

    // 峰值带宽 = 2 (DDR) * clock (MHz) * bus width (bits) / 8 (bytes)
    float peakBandwidth = 2 * memoryClockMHz * (provider.memoryBusWidth() / 8) / 1000.0f;

    return peakBandwidth; // GB/s
}

/**
 * 获取 GPU 理论峰值带宽（默认使用生产环境设备）
 */
inline float getTheoreticalPeakBandwidth() {
    if (!cudaDeviceAvailable()) {
        return 0.0f;
    }
    return getTheoreticalPeakBandwidth(getProductionDeviceInfo());
}

/**
 * 计算效率（相对于理论峰值的百分比）
 */
inline float calculateEfficiency(float actual_gflops, float peak_gflops) {
    if (peak_gflops <= 0)
        return 0.0f;
    return (actual_gflops / peak_gflops) * 100.0f;
}
