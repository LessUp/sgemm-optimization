#pragma once

#include "cuda_utils.cuh"
#include <cuda_runtime.h>

// ============================================================================
// CUDA 性能测量器
// ============================================================================

namespace detail {

/**
 * RAII 包装的 CUDA 事件计时器（内部实现）
 */
class CudaTimer {
  public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    // 禁用拷贝
    CudaTimer(const CudaTimer &) = delete;
    CudaTimer &operator=(const CudaTimer &) = delete;

    // 记录开始事件
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }

    // 记录结束事件并同步
    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }

    // 获取经过的时间（毫秒）
    float elapsedMs() const {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

  private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

} // namespace detail

/**
 * 通用的 GPU 操作性能测量器
 *
 * 支持：
 * - 预热运行（消除首次启动开销）
 * - 多次运行取平均
 * - 自动同步和计时
 */
template <typename RunFunc>
float measureGpuTime(RunFunc func, int warmup_runs = 5, int benchmark_runs = 20) {
    // 预热运行
    for (int i = 0; i < warmup_runs; ++i) {
        func();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时运行
    detail::CudaTimer timer;
    timer.start();
    for (int i = 0; i < benchmark_runs; ++i) {
        func();
    }
    timer.stop();

    return timer.elapsedMs() / benchmark_runs;
}
