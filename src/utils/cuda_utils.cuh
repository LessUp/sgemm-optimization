#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <random>

#include "device_info_provider.cuh"

// ============================================================================
// 命名常量
// ============================================================================

namespace config {
/// 默认 tile 大小（用于 SGEMM 内核）
inline constexpr int kDefaultTileSize = 32;

/// 默认 block 大小（用于 CUDA 内核启动）
inline constexpr int kDefaultBlockSize = 256;

/// 文件名缓冲区大小
inline constexpr int kFilenameBufferSize = 256;
} // namespace config

using config::kDefaultBlockSize;
using config::kDefaultTileSize;
using config::kFilenameBufferSize;

// ============================================================================
// Error Checking Macros
// ============================================================================

#include <stdexcept>
#include <string>

// Exception types for proper RAII cleanup on error
struct CudaError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            throw CudaError(std::string("CUDA error at ") + __FILE__ + ":" +                       \
                            std::to_string(__LINE__) + ": " + cudaGetErrorString(err));            \
        }                                                                                          \
    } while (0)

#define CUBLAS_CHECK(call)                                                                         \
    do {                                                                                           \
        cublasStatus_t status = call;                                                              \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                     \
            throw CudaError(std::string("cuBLAS error at ") + __FILE__ + ":" +                     \
                            std::to_string(__LINE__) + ": code " +                                 \
                            std::to_string(static_cast<int>(status)));                             \
        }                                                                                          \
    } while (0)

#define CURAND_CHECK(call)                                                                         \
    do {                                                                                           \
        curandStatus_t status = call;                                                              \
        if (status != CURAND_STATUS_SUCCESS) {                                                     \
            throw CudaError(std::string("cuRAND error at ") + __FILE__ + ":" +                     \
                            std::to_string(__LINE__) + ": code " +                                 \
                            std::to_string(static_cast<int>(status)));                             \
        }                                                                                          \
    } while (0)

// ============================================================================
// RAII Wrappers for Device Memory
// ============================================================================

template <typename T> class DeviceMemory {
  public:
    DeviceMemory() : ptr_(nullptr), size_(0) {}

    explicit DeviceMemory(size_t count) : ptr_(nullptr), size_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }

    ~DeviceMemory() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    // Move semantics
    DeviceMemory(DeviceMemory &&other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceMemory &operator=(DeviceMemory &&other) noexcept {
        if (this != &other) {
            if (ptr_)
                cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Disable copy
    DeviceMemory(const DeviceMemory &) = delete;
    DeviceMemory &operator=(const DeviceMemory &) = delete;

    T *get() { return ptr_; }
    const T *get() const { return ptr_; }
    size_t size() const { return size_; }

    void copyFromHost(const T *host_ptr, size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copyToHost(T *host_ptr, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host_ptr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void zero() { CUDA_CHECK(cudaMemset(ptr_, 0, size_ * sizeof(T))); }

  private:
    T *ptr_;
    size_t size_;
};

// ============================================================================
// RAII Wrapper for cuBLAS Handle
// ============================================================================

class CublasHandle {
  public:
    CublasHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }

    ~CublasHandle() {
        if (handle_) {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(const CublasHandle &) = delete;
    CublasHandle &operator=(const CublasHandle &) = delete;

    cublasHandle_t get() { return handle_; }

  private:
    cublasHandle_t handle_;
};

// ============================================================================
// Matrix Initialization Functions
// ============================================================================

// Initialize matrix with random values on host
inline void initRandomMatrix(float *data, int rows, int cols, float min_val = -1.0f,
                             float max_val = 1.0f, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);

    for (int i = 0; i < rows * cols; ++i) {
        data[i] = dist(gen);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

// Get GPU device properties
inline void printGPUInfo() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("GPU Device: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM Count: %d\n", prop.multiProcessorCount);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    // Note: memoryClockRate is deprecated in newer CUDA versions
    // Peak bandwidth calculation uses bus width only as approximation
    printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    printf("\n");
}

// ============================================================================
// Device Info Cache - 缓存设备属性避免重复查询
// ============================================================================

/**
 * 设备信息缓存类（单例模式）
 *
 * 缓存 cudaDeviceProp 和常用计算值，避免重复调用 cudaGetDeviceProperties。
 * 首次访问时初始化，之后返回缓存值。
 */
class DeviceInfoCache {
  public:
    /// 获取单例实例
    static DeviceInfoCache &instance() {
        static DeviceInfoCache cache;
        return cache;
    }

    /// 获取缓存的设备属性
    const cudaDeviceProp &prop() const { return prop_; }

    /// 获取设备 ID
    int deviceId() const { return device_; }

    /// 检查是否支持 Tensor Core (sm_70+)
    bool hasTensorCores() const { return prop_.major >= 7; }

    /// 获取每个 SM 的 CUDA 核心数（基于架构）
    int coresPerSM() const { return coresPerSM_; }

    /// 获取时钟频率 (GHz)
    float clockGHz() const { return clockGHz_; }

  private:
    DeviceInfoCache() {
        CUDA_CHECK(cudaGetDevice(&device_));
        CUDA_CHECK(cudaGetDeviceProperties(&prop_, device_));
        coresPerSM_ = computeCoresPerSM();
        clockGHz_ = static_cast<float>(prop_.clockRate) / 1e6f;
    }

    int computeCoresPerSM() const {
        // 参考: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
        if (prop_.major == 7) {
            return 64; // Volta (sm_70, sm_72), Turing (sm_75)
        } else if (prop_.major == 8) {
            return (prop_.minor == 0 || prop_.minor == 6) ? 64 : 128; // A100/sm_80, A10G/sm_86: 64
        } else if (prop_.major == 9) {
            return 128; // Hopper (sm_90)
        }
        return 64; // 默认回退
    }

    int device_;
    cudaDeviceProp prop_;
    int coresPerSM_;
    float clockGHz_;
};

// ============================================================================
// Device Info Provider Implementation
// ============================================================================

/**
 * Production adapter implementation: queries from DeviceInfoCache
 */
inline DeviceInfoProvider ProductionDeviceInfoProvider::get() const {
    DeviceInfoCache &cache = DeviceInfoCache::instance();
    return DeviceInfoProvider{
        &cache.prop(),
        cache.coresPerSM(),
        cache.clockGHz(),
    };
}

/**
 * Convenience function: get production device info
 */
inline DeviceInfoProvider getProductionDeviceInfo() {
    return ProductionDeviceInfoProvider{}.get();
}
