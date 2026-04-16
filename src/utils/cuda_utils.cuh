#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <random>

// ============================================================================
// Error Checking Macros
// ============================================================================

#include <stdexcept>
#include <string>

// Exception types for proper RAII cleanup on error
struct CudaError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      throw CudaError(std::string("CUDA error at ") + __FILE__ + ":" +         \
                      std::to_string(__LINE__) + ": " +                        \
                      cudaGetErrorString(err));                                \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      throw CudaError(std::string("cuBLAS error at ") + __FILE__ + ":" +       \
                      std::to_string(__LINE__) + ": code " +                   \
                      std::to_string(static_cast<int>(status)));               \
    }                                                                          \
  } while (0)

#define CURAND_CHECK(call)                                                     \
  do {                                                                         \
    curandStatus_t status = call;                                              \
    if (status != CURAND_STATUS_SUCCESS) {                                     \
      throw CudaError(std::string("cuRAND error at ") + __FILE__ + ":" +       \
                      std::to_string(__LINE__) + ": code " +                   \
                      std::to_string(static_cast<int>(status)));               \
    }                                                                          \
  } while (0)

// ============================================================================
// RAII Wrappers for Device Memory
// ============================================================================

template <typename T> class DeviceMemory {
public:
  DeviceMemory() : ptr_(nullptr), size_(0) {}

  explicit DeviceMemory(size_t count) : size_(count) {
    CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
  }

  ~DeviceMemory() {
    if (ptr_) {
      cudaFree(ptr_);
    }
  }

  // Move semantics
  DeviceMemory(DeviceMemory &&other) noexcept
      : ptr_(other.ptr_), size_(other.size_) {
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
    CUDA_CHECK(
        cudaMemcpy(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
  }

  void copyToHost(T *host_ptr, size_t count) const {
    CUDA_CHECK(
        cudaMemcpy(host_ptr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
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
inline void initRandomMatrix(float *data, int rows, int cols,
                             float min_val = -1.0f, float max_val = 1.0f,
                             unsigned int seed = 42) {
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
  printf("  Global Memory: %.2f GB\n",
         prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
  // Note: memoryClockRate is deprecated in newer CUDA versions
  // Peak bandwidth calculation uses bus width only as approximation
  printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
  printf("\n");
}
