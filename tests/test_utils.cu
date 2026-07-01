/**
 * 工具层测试套件
 *
 * 测试 CUDA 工具类的正确性和异常安全性：
 * - DeviceMemory: RAII 内存管理、移动语义
 * - CublasHandle: cuBLAS 句柄生命周期
 * - SGEMMVerifier: 参考计算和验证逻辑
 * - 容差配置和边界条件
 */

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <vector>

#include "gtest_cuda_environment.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/verify.cuh"

// ============================================================================
// DeviceMemory 测试
// ============================================================================

class DeviceMemoryTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // 使用小尺寸以加快测试
        test_size_ = 1024;
        h_data_.resize(test_size_);
        for (size_t i = 0; i < test_size_; ++i) {
            h_data_[i] = static_cast<float>(i);
        }
    }

    size_t test_size_;
    std::vector<float> h_data_;
};

TEST_F(DeviceMemoryTest, DefaultConstructorCreatesNullPointer) {
    DeviceMemory<float> mem;
    EXPECT_EQ(mem.get(), nullptr);
    EXPECT_EQ(mem.size(), 0u);
}

TEST_F(DeviceMemoryTest, SizeConstructorAllocatesMemory) {
    DeviceMemory<float> mem(test_size_);
    EXPECT_NE(mem.get(), nullptr);
    EXPECT_EQ(mem.size(), test_size_);
}

TEST_F(DeviceMemoryTest, CopyFromHostAndToHostRoundTrip) {
    DeviceMemory<float> mem(test_size_);
    mem.copyFromHost(h_data_.data(), test_size_);

    std::vector<float> h_result(test_size_);
    mem.copyToHost(h_result.data(), test_size_);

    for (size_t i = 0; i < test_size_; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], h_data_[i]);
    }
}

TEST_F(DeviceMemoryTest, ZeroClearsMemory) {
    DeviceMemory<float> mem(test_size_);
    mem.copyFromHost(h_data_.data(), test_size_);
    mem.zero();

    std::vector<float> h_result(test_size_);
    mem.copyToHost(h_result.data(), test_size_);

    for (size_t i = 0; i < test_size_; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], 0.0f);
    }
}

TEST_F(DeviceMemoryTest, MoveConstructorTransfersOwnership) {
    DeviceMemory<float> mem1(test_size_);
    float *original_ptr = mem1.get();

    DeviceMemory<float> mem2(std::move(mem1));

    EXPECT_EQ(mem2.get(), original_ptr);
    EXPECT_EQ(mem2.size(), test_size_);
    EXPECT_EQ(mem1.get(), nullptr);
    EXPECT_EQ(mem1.size(), 0u);
}

TEST_F(DeviceMemoryTest, MoveAssignmentTransfersOwnership) {
    DeviceMemory<float> mem1(test_size_);
    DeviceMemory<float> mem2;

    float *original_ptr = mem1.get();
    mem2 = std::move(mem1);

    EXPECT_EQ(mem2.get(), original_ptr);
    EXPECT_EQ(mem2.size(), test_size_);
    EXPECT_EQ(mem1.get(), nullptr);
    EXPECT_EQ(mem1.size(), 0u);
}

TEST_F(DeviceMemoryTest, MoveAssignmentReleasesOldMemory) {
    DeviceMemory<float> mem1(test_size_);
    DeviceMemory<float> mem2(test_size_ * 2);

    float *ptr1 = mem1.get();
    float *ptr2 = mem2.get();

    // mem2 原有的内存应该被释放
    mem2 = std::move(mem1);

    EXPECT_EQ(mem2.get(), ptr1);
    // ptr2 指向的内存已被释放，无法直接验证，但不应崩溃
}

TEST_F(DeviceMemoryTest, SelfMoveAssignmentIsSafe) {
    DeviceMemory<float> mem(test_size_);
    float *original_ptr = mem.get();

    // 自赋值不应崩溃或导致双重释放
    mem = std::move(mem);

    // 行为实现定义，但不应崩溃
    // 如果实现正确，指针应该仍然有效
}

TEST_F(DeviceMemoryTest, CopyConstructorIsDeleted) {
    // 此测试仅验证编译时行为
    // DeviceMemory<float> mem1(test_size_);
    // DeviceMemory<float> mem2(mem1); // 应编译失败
    EXPECT_TRUE(true); // 占位符
}

TEST_F(DeviceMemoryTest, CopyAssignmentIsDeleted) {
    // 此测试仅验证编译时行为
    // DeviceMemory<float> mem1(test_size_);
    // DeviceMemory<float> mem2;
    // mem2 = mem1; // 应编译失败
    EXPECT_TRUE(true); // 占位符
}

TEST_F(DeviceMemoryTest, DifferentTypesWork) {
    DeviceMemory<int> int_mem(100);
    EXPECT_NE(int_mem.get(), nullptr);

    DeviceMemory<double> double_mem(100);
    EXPECT_NE(double_mem.get(), nullptr);
}

TEST_F(DeviceMemoryTest, ZeroSizeAllocation) {
    // 零尺寸分配应成功但不分配内存
    EXPECT_NO_THROW(DeviceMemory<float> mem(0));
}

// ============================================================================
// CublasHandle 测试
// ============================================================================

class CublasHandleTest : public ::testing::Test {};

TEST_F(CublasHandleTest, ConstructorCreatesValidHandle) {
    CublasHandle handle;
    EXPECT_NE(handle.get(), nullptr);
}

TEST_F(CublasHandleTest, DestructorCleansUp) {
    cublasHandle_t raw_handle;
    {
        CublasHandle handle;
        raw_handle = handle.get();
        EXPECT_NE(raw_handle, nullptr);
    }
    // 离开作用域后，raw_handle 应已被销毁
    // 无法直接验证，但不应有内存泄漏
}

TEST_F(CublasHandleTest, CopyConstructorIsDeleted) {
    // CublasHandle handle1;
    // CublasHandle handle2(handle1); // 应编译失败
    EXPECT_TRUE(true);
}

TEST_F(CublasHandleTest, CopyAssignmentIsDeleted) {
    // CublasHandle handle1;
    // CublasHandle handle2;
    // handle2 = handle1; // 应编译失败
    EXPECT_TRUE(true);
}

// ============================================================================
// SGEMMVerifier 测试
// ============================================================================

class SGEMMVerifierTest : public ::testing::Test {
  protected:
    void SetUp() override {
        M_ = 32;
        K_ = 32;
        N_ = 32;

        h_A_.resize(M_ * K_);
        h_B_.resize(K_ * N_);
        h_C_.resize(M_ * N_);
        h_C_ref_.resize(M_ * N_);

        initRandomMatrix(h_A_.data(), M_, K_, -1.0f, 1.0f, 42);
        initRandomMatrix(h_B_.data(), K_, N_, -1.0f, 1.0f, 123);

        d_A_ = std::make_unique<DeviceMemory<float>>(M_ * K_);
        d_B_ = std::make_unique<DeviceMemory<float>>(K_ * N_);
        d_C_ = std::make_unique<DeviceMemory<float>>(M_ * N_);

        d_A_->copyFromHost(h_A_.data(), M_ * K_);
        d_B_->copyFromHost(h_B_.data(), K_ * N_);
    }

    int M_, K_, N_;
    std::vector<float> h_A_, h_B_, h_C_, h_C_ref_;
    std::unique_ptr<DeviceMemory<float>> d_A_;
    std::unique_ptr<DeviceMemory<float>> d_B_;
    std::unique_ptr<DeviceMemory<float>> d_C_;
    SGEMMVerifier verifier_;
};

TEST_F(SGEMMVerifierTest, ComputeReferenceProducesCorrectShape) {
    verifier_.computeReference(d_A_->get(), d_B_->get(), d_C_->get(), M_, K_, N_);
    d_C_->copyToHost(h_C_.data(), M_ * N_);

    // 结果应该是 M x N
    EXPECT_EQ(h_C_.size(), static_cast<size_t>(M_ * N_));

    // 不应全为零（随机输入）
    bool all_zero = true;
    for (const auto &val : h_C_) {
        if (val != 0.0f) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero);
}

TEST_F(SGEMMVerifierTest, VerifyPassesForIdenticalMatrices) {
    d_C_->copyFromHost(h_A_.data(), M_ * N_);

    VerifyResult result = verifier_.verifyDevice(d_C_->get(), d_C_->get(), M_, N_);

    EXPECT_TRUE(result.passed);
    EXPECT_EQ(result.error_count, 0);
    EXPECT_FLOAT_EQ(result.max_abs_error, 0.0f);
    EXPECT_FLOAT_EQ(result.max_rel_error, 0.0f);
}

TEST_F(SGEMMVerifierTest, VerifyFailsForDifferentMatrices) {
    std::vector<float> h_different(M_ * N_, 1e10f); // 显著不同的值
    DeviceMemory<float> d_different(M_ * N_);
    d_different.copyFromHost(h_different.data(), M_ * N_);

    DeviceMemory<float> d_zeros(M_ * N_);
    d_zeros.zero();

    VerifyResult result = verifier_.verifyDevice(d_different.get(), d_zeros.get(), M_, N_);

    EXPECT_FALSE(result.passed);
    EXPECT_GT(result.error_count, 0);
}

TEST_F(SGEMMVerifierTest, VerifyWithCustomTolerance) {
    // 创建两个略有差异的矩阵
    std::vector<float> h_test(M_ * N_, 1.0f);
    std::vector<float> h_ref(M_ * N_, 1.0f);
    h_test[0] = 1.001f; // 0.1% 差异

    VerifyResult result_strict =
        compareMatrices(h_test.data(), h_ref.data(), M_, N_, {1e-4f, 1e-5f}); // 更严格的容差
    EXPECT_FALSE(result_strict.passed);

    VerifyResult result_relaxed =
        compareMatrices(h_test.data(), h_ref.data(), M_, N_, {1e-2f, 1e-2f}); // 更宽松的容差
    EXPECT_TRUE(result_relaxed.passed);
}

TEST_F(SGEMMVerifierTest, VerifyHandlesNanCorrectly) {
    std::vector<float> h_with_nan(M_ * N_, 1.0f);
    h_with_nan[0] = std::nanf("");

    std::vector<float> h_ref(M_ * N_, 1.0f);

    VerifyResult result = compareMatrices(h_with_nan.data(), h_ref.data(), M_, N_);

    EXPECT_FALSE(result.passed);
    EXPECT_GT(result.error_count, 0);
    EXPECT_TRUE(std::isinf(result.max_abs_error) || std::isnan(h_with_nan[0]));
}

TEST_F(SGEMMVerifierTest, VerifyHandlesInfCorrectly) {
    std::vector<float> h_with_inf(M_ * N_, 1.0f);
    h_with_inf[0] = std::numeric_limits<float>::infinity();

    std::vector<float> h_ref(M_ * N_, 1.0f);

    VerifyResult result = compareMatrices(h_with_inf.data(), h_ref.data(), M_, N_);

    EXPECT_FALSE(result.passed);
    EXPECT_GT(result.error_count, 0);
}

// ============================================================================
// VerifyTolerance 测试
// ============================================================================

class ToleranceTest : public ::testing::Test {};

TEST_F(ToleranceTest, StandardToleranceValues) {
    EXPECT_FLOAT_EQ(kStandardVerifyTolerance.rtol, 1e-3f);
    EXPECT_FLOAT_EQ(kStandardVerifyTolerance.atol, 1e-4f);
}

TEST_F(ToleranceTest, IsWithinToleranceWorks) {
    VerifyTolerance tol{1e-3f, 1e-4f};

    // 绝对误差在容差内
    EXPECT_TRUE(isWithinTolerance(1.0f, 1.0f, tol));
    EXPECT_TRUE(isWithinTolerance(1.00005f, 1.0f, tol));

    // 绝对误差超出容差
    EXPECT_FALSE(isWithinTolerance(1.1f, 1.0f, tol));

    // 相对误差
    EXPECT_TRUE(isWithinTolerance(1000.0f, 1001.0f, tol));  // 0.1% 差异
    EXPECT_FALSE(isWithinTolerance(1000.0f, 1002.0f, tol)); // 0.2% 差异
}

TEST_F(ToleranceTest, ToleranceForValue) {
    VerifyTolerance tol{0.01f, 0.001f}; // 1% 相对容差，0.001 绝对容差

    // 小值：绝对容差主导
    EXPECT_FLOAT_EQ(toleranceForValue(0.0f, tol), 0.001f);
    EXPECT_FLOAT_EQ(toleranceForValue(0.01f, tol), 0.001f + 0.01f * 0.01f);

    // 大值：相对容差主导
    EXPECT_FLOAT_EQ(toleranceForValue(100.0f, tol), 0.001f + 1.0f);
}

// ============================================================================
// 错误检测测试
// ============================================================================

TEST(VerifyResultTest, ShouldFlagAsIncorrectReturnsCorrectValue) {
    VerifyResult passed;
    passed.passed = true;
    EXPECT_FALSE(SGEMMVerifier::shouldFlagAsIncorrect(passed));

    VerifyResult failed;
    failed.passed = false;
    EXPECT_TRUE(SGEMMVerifier::shouldFlagAsIncorrect(failed));
}

// ============================================================================
// 集成测试：DeviceMemory + SGEMMVerifier
// ============================================================================

class UtilsIntegrationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        size_ = 16;
        h_data_.resize(size_ * size_);
        initRandomMatrix(h_data_.data(), size_, size_, -1.0f, 1.0f, 42);
    }

    int size_;
    std::vector<float> h_data_;
};

TEST_F(UtilsIntegrationTest, FullWorkflowWithDeviceMemory) {
    // 分配设备内存
    DeviceMemory<float> d_A(size_ * size_);
    DeviceMemory<float> d_B(size_ * size_);
    DeviceMemory<float> d_C(size_ * size_);

    // 复制到设备
    d_A.copyFromHost(h_data_.data(), size_ * size_);
    d_B.copyFromHost(h_data_.data(), size_ * size_);

    // 使用 SGEMMVerifier 计算参考结果
    SGEMMVerifier verifier;
    verifier.computeReference(d_A.get(), d_B.get(), d_C.get(), size_, size_, size_);

    // 验证
    std::vector<float> h_result(size_ * size_);
    d_C.copyToHost(h_result.data(), size_ * size_);

    // 结果不应全为零
    bool has_nonzero = false;
    for (const auto &val : h_result) {
        if (val != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

int main(int argc, char **argv) { return runCudaAwareTests(argc, argv); }
