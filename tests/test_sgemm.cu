/**
 * SGEMM Optimization Test Suite
 *
 * Property-based tests for verifying correctness of all SGEMM kernel
 * implementations. Uses Google Test framework with parameterized tests for
 * comprehensive coverage.
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <tuple>
#include <vector>

#include "kernels/bank_conflict_free_sgemm.cuh"
#include "kernels/double_buffer_sgemm.cuh"
#include "kernels/naive_sgemm.cuh"
#include "kernels/tensor_core_sgemm.cuh"
#include "kernels/tiled_sgemm.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/verify.cuh"

namespace {
constexpr int PBT_ITERATIONS = 100;

std::vector<std::tuple<int, int, int>> getStandardDimensions() {
  return {
      {1, 1, 1},       {16, 16, 16},    {32, 32, 32},     {64, 64, 64},
      {128, 128, 128}, {256, 256, 256}, {512, 512, 512},  {64, 128, 256},
      {256, 64, 128},  {128, 256, 64},  {511, 513, 1025},
  };
}

std::vector<std::tuple<int, int, int>> getTensorCoreFastPathDimensions() {
  return {
      {16, 16, 16},    {32, 32, 32},   {64, 64, 64},   {128, 128, 128},
      {256, 256, 256}, {64, 128, 256}, {256, 64, 128},
  };
}

std::vector<std::tuple<int, int, int>> getTensorCoreFallbackDimensions() {
  return {
      {17, 16, 16},
      {31, 48, 17},
      {33, 33, 33},
      {511, 513, 1025},
  };
}

void computeReference(cublasHandle_t handle, const float *d_A, const float *d_B,
                      float *d_C, int M, int K, int N) {
  float alpha = 1.0f;
  float beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           d_B, N, d_A, K, &beta, d_C, N));
}
} // namespace

class ErrorDetectionTest : public ::testing::Test {
protected:
  void SetUp() override {
    size_ = 64 * 64;
    h_test_.resize(size_);
    h_ref_.resize(size_);
  }

  std::vector<float> h_test_;
  std::vector<float> h_ref_;
  size_t size_;
};

TEST_F(ErrorDetectionTest, StandardKernelErrorDetection) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int iter = 0; iter < PBT_ITERATIONS; ++iter) {
    for (size_t i = 0; i < size_; ++i) {
      h_ref_[i] = dist(gen);
    }

    float error_magnitude = 1e-2f;
    for (size_t i = 0; i < size_; ++i) {
      h_test_[i] = h_ref_[i] + error_magnitude * (dist(gen) > 0 ? 1 : -1);
    }

    VerifyResult result = compareMatrices(h_test_.data(), h_ref_.data(), 64, 64,
                                          kStandardVerifyTolerance);

    EXPECT_TRUE(SGEMMVerifier::shouldFlagAsIncorrect(result))
        << "Iteration " << iter << ": error above tolerance should be flagged";
  }
}

TEST_F(ErrorDetectionTest, StandardKernelPassesWithinTolerance) {
  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int iter = 0; iter < PBT_ITERATIONS; ++iter) {
    for (size_t i = 0; i < size_; ++i) {
      h_ref_[i] = dist(gen);
    }

    float error_magnitude = 1e-6f;
    for (size_t i = 0; i < size_; ++i) {
      h_test_[i] = h_ref_[i] + error_magnitude * dist(gen);
    }

    VerifyResult result = compareMatrices(h_test_.data(), h_ref_.data(), 64, 64,
                                          kStandardVerifyTolerance);

    EXPECT_TRUE(result.passed)
        << "Iteration " << iter << ": error within tolerance should pass";
  }
}

TEST_F(ErrorDetectionTest, TensorCoreErrorDetection) {
  std::mt19937 gen(456);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int iter = 0; iter < PBT_ITERATIONS; ++iter) {
    for (size_t i = 0; i < size_; ++i) {
      h_ref_[i] = dist(gen);
    }

    float error_magnitude = 5e-2f;
    for (size_t i = 0; i < size_; ++i) {
      h_test_[i] = h_ref_[i] + error_magnitude * (dist(gen) > 0 ? 1 : -1);
    }

    VerifyResult result = compareMatrices(h_test_.data(), h_ref_.data(), 64, 64,
                                          kTensorCoreVerifyTolerance);

    EXPECT_TRUE(SGEMMVerifier::shouldFlagAsIncorrect(result))
        << "Iteration " << iter
        << ": Tensor Core error above tolerance should be flagged";
  }
}

class SGEMMKernelTest
    : public ::testing::TestWithParam<std::tuple<int, int, int>> {
protected:
  void SetUp() override {
    std::tie(M_, K_, N_) = GetParam();

    h_A_.resize(M_ * K_);
    h_B_.resize(K_ * N_);
    h_C_.resize(M_ * N_);
    h_C_ref_.resize(M_ * N_);

    initRandomMatrix(h_A_.data(), M_, K_, -1.0f, 1.0f, 42);
    initRandomMatrix(h_B_.data(), K_, N_, -1.0f, 1.0f, 123);

    CUDA_CHECK(cudaMalloc(&d_A_, M_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_, K_ * N_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_, M_ * N_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_ref_, M_ * N_ * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A_, h_A_.data(), M_ * K_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_, h_B_.data(), K_ * N_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    verifier_.computeReference(d_A_, d_B_, d_C_ref_, M_, K_, N_);
    CUDA_CHECK(cudaMemcpy(h_C_ref_.data(), d_C_ref_, M_ * N_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  void TearDown() override {
    if (d_A_)
      cudaFree(d_A_);
    if (d_B_)
      cudaFree(d_B_);
    if (d_C_)
      cudaFree(d_C_);
    if (d_C_ref_)
      cudaFree(d_C_ref_);
  }

  template <typename LaunchFn>
  VerifyResult
  runKernelAndCompare(LaunchFn launch_fn,
                      VerifyTolerance tolerance = kStandardVerifyTolerance) {
    CUDA_CHECK(cudaMemset(d_C_, 0, M_ * N_ * sizeof(float)));
    launch_fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_.data(), d_C_, M_ * N_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return compareMatrices(h_C_.data(), h_C_ref_.data(), M_, N_, tolerance);
  }

  int M_, K_, N_;
  std::vector<float> h_A_, h_B_, h_C_, h_C_ref_;
  float *d_A_ = nullptr;
  float *d_B_ = nullptr;
  float *d_C_ = nullptr;
  float *d_C_ref_ = nullptr;
  SGEMMVerifier verifier_;
};

class NaiveSGEMMTest : public SGEMMKernelTest {};

TEST_P(NaiveSGEMMTest, CorrectnessProperty) {
  VerifyResult result = runKernelAndCompare(
      [&] { launch_naive_sgemm<>(d_A_, d_B_, d_C_, M_, K_, N_); });

  EXPECT_TRUE(result.passed)
      << "Naive SGEMM failed for dimensions " << M_ << "x" << K_ << "x" << N_
      << " (max_rel_error: " << result.max_rel_error << ")";
}

INSTANTIATE_TEST_SUITE_P(StandardDimensions, NaiveSGEMMTest,
                         ::testing::ValuesIn(getStandardDimensions()));

class TiledSGEMMTest : public SGEMMKernelTest {};

TEST_P(TiledSGEMMTest, CorrectnessProperty) {
  VerifyResult result = runKernelAndCompare(
      [&] { launch_tiled_sgemm<32>(d_A_, d_B_, d_C_, M_, K_, N_); });

  EXPECT_TRUE(result.passed)
      << "Tiled SGEMM failed for dimensions " << M_ << "x" << K_ << "x" << N_
      << " (max_rel_error: " << result.max_rel_error << ")";
}

INSTANTIATE_TEST_SUITE_P(StandardDimensions, TiledSGEMMTest,
                         ::testing::ValuesIn(getStandardDimensions()));

class BankConflictFreeSGEMMTest : public SGEMMKernelTest {};

TEST_P(BankConflictFreeSGEMMTest, CorrectnessProperty) {
  VerifyResult result = runKernelAndCompare([&] {
    launch_bank_conflict_free_sgemm<32>(d_A_, d_B_, d_C_, M_, K_, N_);
  });

  EXPECT_TRUE(result.passed)
      << "BankConflictFree SGEMM failed for dimensions " << M_ << "x" << K_
      << "x" << N_ << " (max_rel_error: " << result.max_rel_error << ")";
}

INSTANTIATE_TEST_SUITE_P(StandardDimensions, BankConflictFreeSGEMMTest,
                         ::testing::ValuesIn(getStandardDimensions()));

class DoubleBufferSGEMMTest : public SGEMMKernelTest {};

TEST_P(DoubleBufferSGEMMTest, CorrectnessProperty) {
  VerifyResult result = runKernelAndCompare(
      [&] { launch_double_buffer_sgemm<32>(d_A_, d_B_, d_C_, M_, K_, N_); });

  EXPECT_TRUE(result.passed)
      << "DoubleBuffer SGEMM failed for dimensions " << M_ << "x" << K_ << "x"
      << N_ << " (max_rel_error: " << result.max_rel_error << ")";
}

INSTANTIATE_TEST_SUITE_P(StandardDimensions, DoubleBufferSGEMMTest,
                         ::testing::ValuesIn(getStandardDimensions()));

class TensorCoreSGEMMTest : public SGEMMKernelTest {};

TEST_P(TensorCoreSGEMMTest, FastPathCorrectnessProperty) {
  if (!tensorCoresAvailable()) {
    GTEST_SKIP() << "Tensor Cores require sm_70 or higher";
  }

  ASSERT_TRUE(tensorCoreDimensionsSupported(M_, K_, N_));

  VerifyResult result = runKernelAndCompare(
      [&] { launch_tensor_core_sgemm(d_A_, d_B_, d_C_, M_, K_, N_); },
      kTensorCoreVerifyTolerance);

  EXPECT_TRUE(result.passed)
      << "TensorCore SGEMM fast path failed for dimensions " << M_ << "x" << K_
      << "x" << N_ << " (max_rel_error: " << result.max_rel_error << ")";
}

INSTANTIATE_TEST_SUITE_P(
    TensorCoreFastPathDimensions, TensorCoreSGEMMTest,
    ::testing::ValuesIn(getTensorCoreFastPathDimensions()));

class TensorCoreFallbackTest : public SGEMMKernelTest {};

TEST_P(TensorCoreFallbackTest, NonAlignedInputsFallbackSafely) {
  VerifyResult result = runKernelAndCompare(
      [&] { launch_tensor_core_sgemm(d_A_, d_B_, d_C_, M_, K_, N_); },
      kStandardVerifyTolerance);

  EXPECT_TRUE(result.passed)
      << "TensorCore fallback failed for dimensions " << M_ << "x" << K_ << "x"
      << N_ << " (max_rel_error: " << result.max_rel_error << ")";
}

INSTANTIATE_TEST_SUITE_P(
    TensorCoreFallbackDimensions, TensorCoreFallbackTest,
    ::testing::ValuesIn(getTensorCoreFallbackDimensions()));

TEST(TensorCoreWrapperTest, ZeroSizeInputsReturnSafely) {
  EXPECT_NO_THROW(
      launch_tensor_core_sgemm(nullptr, nullptr, nullptr, 0, 16, 16));
  EXPECT_NO_THROW(
      launch_tensor_core_sgemm(nullptr, nullptr, nullptr, 16, 0, 16));
  EXPECT_NO_THROW(
      launch_tensor_core_sgemm(nullptr, nullptr, nullptr, 16, 16, 0));
}

class DimensionInvarianceTest : public ::testing::Test {
protected:
  void SetUp() override { CUBLAS_CHECK(cublasCreate(&handle_)); }

  void TearDown() override { cublasDestroy(handle_); }

  cublasHandle_t handle_;
};

TEST_F(DimensionInvarianceTest, AllStandardKernelsWorkWithVariousDimensions) {
  std::mt19937 gen(789);
  std::uniform_int_distribution<int> dist(1, 16);

  for (int iter = 0; iter < PBT_ITERATIONS; ++iter) {
    int M = dist(gen) * 32;
    int K = dist(gen) * 32;
    int N = dist(gen) * 32;

    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_ref(M * N);
    initRandomMatrix(h_A.data(), M, K, -1.0f, 1.0f, iter);
    initRandomMatrix(h_B.data(), K, N, -1.0f, 1.0f, iter + 1000);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    float *d_ref = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float),
                          cudaMemcpyHostToDevice));

    computeReference(handle_, d_A, d_B, d_ref, M, K, N);
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    auto testKernel = [&](const char *name, auto kernel_func) {
      CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
      kernel_func(d_A, d_B, d_C, M, K, N);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float),
                            cudaMemcpyDeviceToHost));

      VerifyResult result = compareMatrices(h_C.data(), h_ref.data(), M, N,
                                            kStandardVerifyTolerance);
      EXPECT_TRUE(result.passed)
          << name << " failed at iteration " << iter << " with dimensions " << M
          << "x" << K << "x" << N;
    };

    testKernel("Naive", launch_naive_sgemm<>);
    testKernel("Tiled", launch_tiled_sgemm<32>);
    testKernel("BankConflictFree", launch_bank_conflict_free_sgemm<32>);
    testKernel("DoubleBuffer", launch_double_buffer_sgemm<32>);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_ref);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  printGPUInfo();
  return RUN_ALL_TESTS();
}
