/**
 * 性能回归测试框架
 *
 * 为每个 SGEMM 内核定义最小性能阈值，检测性能退化。
 * 阈值基于理论峰值的百分比，保证各内核达到预期效率。
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "kernels/bank_conflict_free_sgemm.cuh"
#include "kernels/double_buffer_sgemm.cuh"
#include "kernels/naive_sgemm.cuh"
#include "kernels/tensor_core_fallback.cuh"
#include "kernels/tensor_core_sgemm.cuh"
#include "kernels/tiled_sgemm.cuh"
#include "utils/benchmark_core.cuh"
#include "utils/benchmark_metrics.cuh"
#include "utils/cuda_utils.cuh"

// ============================================================================
// 性能回归测试
// ============================================================================

class PerformanceRegressionTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // 获取当前 GPU 信息
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop_, device));

        // 计算理论峰值
        peak_gflops_ = getTheoreticalPeakGflops();
        peak_bandwidth_ = getTheoreticalPeakBandwidth();

        // 性能效率阈值（相对于理论峰值的百分比）
        // 这些是保守值，实际性能可能更高
        min_efficiency_ = {
            {"Naive", 0.05f},            // 5% 峰值
            {"Tiled", 0.20f},            // 20% 峰值
            {"BankConflictFree", 0.30f}, // 30% 峰值
            {"DoubleBuffer", 0.35f},     // 35% 峰值
            {"TensorCore", 0.50f}        // 50% 峰值（当可用时）
        };

        // 测试维度
        test_dimensions_ = {
            {512, 512, 512},
            {1024, 1024, 1024},
        };
    }

    // 测量内核性能
    template <typename LaunchFunc>
    float measureGflops(LaunchFunc launch_func, int M, int K, int N) {
        // 分配内存
        DeviceMemory<float> d_A(M * K);
        DeviceMemory<float> d_B(K * N);
        DeviceMemory<float> d_C(M * N);

        // 初始化
        std::vector<float> h_A(M * K, 1.0f);
        std::vector<float> h_B(K * N, 1.0f);
        d_A.copyFromHost(h_A.data(), M * K);
        d_B.copyFromHost(h_B.data(), K * N);

        // 测量
        float time_ms =
            measureGpuTime([&]() { launch_func(d_A.get(), d_B.get(), d_C.get(), M, K, N); }, 3, 10);

        // 计算 GFLOPS
        PerformanceMetrics metrics = calculateSgemmMetrics(M, K, N, time_ms);
        return metrics.gflops;
    }

    // 运行性能测试
    void runPerformanceTest(const std::string &kernel_name, auto launch_func, int M, int K, int N,
                            VerifyTolerance tolerance = kStandardVerifyTolerance) {
        float gflops = measureGflops(launch_func, M, K, N);

        // 计算最小可接受 GFLOPS
        float threshold = min_efficiency_.count(kernel_name) ? min_efficiency_[kernel_name] : 0.1f;
        float min_gflops = peak_gflops_ * threshold;

        // 输出结果
        printf("  %-25s | %4d x %4d x %4d | %10.2f GFLOPS | min: %8.2f | %s\n", kernel_name.c_str(),
               M, K, N, gflops, min_gflops, gflops >= min_gflops ? "PASS" : "REGRESSION");

        // 检查性能回归
        EXPECT_GE(gflops, min_gflops)
            << kernel_name << " performance regression detected!\n"
            << "  Actual: " << gflops << " GFLOPS\n"
            << "  Minimum: " << min_gflops << " GFLOPS (" << (threshold * 100) << "% of peak)\n"
            << "  GPU: " << prop_.name << " (" << peak_gflops_ << " GFLOPS peak)";
    }

    cudaDeviceProp prop_;
    float peak_gflops_;
    float peak_bandwidth_;
    std::unordered_map<std::string, float> min_efficiency_;
    std::vector<std::tuple<int, int, int>> test_dimensions_;
};

// ============================================================================
// 测试用例
// ============================================================================

TEST_F(PerformanceRegressionTest, NaiveKernelPerformance) {
    printf("\nNaive Kernel Performance:\n");
    for (const auto &[M, K, N] : test_dimensions_) {
        runPerformanceTest("Naive", launch_naive_sgemm<>, M, K, N);
    }
}

TEST_F(PerformanceRegressionTest, TiledKernelPerformance) {
    printf("\nTiled Kernel Performance:\n");
    for (const auto &[M, K, N] : test_dimensions_) {
        runPerformanceTest("Tiled", launch_tiled_sgemm<32>, M, K, N);
    }
}

TEST_F(PerformanceRegressionTest, BankConflictFreeKernelPerformance) {
    printf("\nBank-Conflict-Free Kernel Performance:\n");
    for (const auto &[M, K, N] : test_dimensions_) {
        runPerformanceTest("BankConflictFree", launch_bank_conflict_free_sgemm<32>, M, K, N);
    }
}

TEST_F(PerformanceRegressionTest, DoubleBufferKernelPerformance) {
    printf("\nDouble-Buffer Kernel Performance:\n");
    for (const auto &[M, K, N] : test_dimensions_) {
        runPerformanceTest("DoubleBuffer", launch_double_buffer_sgemm<32>, M, K, N);
    }
}

TEST_F(PerformanceRegressionTest, TensorCoreKernelPerformance) {
    if (!tensorCoresAvailable()) {
        GTEST_SKIP() << "Tensor Cores require sm_70 or higher";
    }

    printf("\nTensor Core Kernel Performance:\n");
    // Tensor Core 需要对齐维度
    std::vector<std::tuple<int, int, int>> tc_dimensions = {
        {128, 128, 128},
        {256, 256, 256},
    };

    for (const auto &[M, K, N] : tc_dimensions) {
        runPerformanceTest(
            "TensorCore",
            [](const float *A, const float *B, float *C, int M, int K, int N, cudaStream_t s) {
                launch_tensor_core_sgemm_with_fallback(A, B, C, M, K, N,
                                                       defaultTensorCoreFallback(), s);
            },
            M, K, N, kTensorCoreVerifyTolerance);
    }
}

// ============================================================================
// 峰值性能测试
// ============================================================================

TEST_F(PerformanceRegressionTest, PeakPerformanceReference) {
    // 此测试验证理论峰值计算是否合理
    printf("\nGPU Performance Reference:\n");
    printf("  GPU: %s\n", prop_.name);
    printf("  Compute Capability: %d.%d\n", prop_.major, prop_.minor);
    printf("  SM Count: %d\n", prop_.multiProcessorCount);
    printf("  Theoretical Peak FP32: %.2f GFLOPS\n", peak_gflops_);
    printf("  Theoretical Peak Bandwidth: %.2f GB/s\n", peak_bandwidth_);

    // 验证峰值在合理范围内
    EXPECT_GT(peak_gflops_, 1000.0f) << "Peak GFLOPS seems too low";
    EXPECT_LT(peak_gflops_, 100000.0f) << "Peak GFLOPS seems too high";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    printGPUInfo();
    return RUN_ALL_TESTS();
}
