/**
 * Kernel catalog module test suite
 *
 * Tests the kernel catalog registration system:
 * - Catalog contains expected kernels
 * - Entries have valid names and launchers
 * - Launch functions are callable without crashes
 * - Order preservation of kernel entries
 */

#include <gtest/gtest.h>
#include "kernels/kernel_catalog.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/benchmark_settings.cuh"

// ============================================================================
// Kernel Catalog Tests
// ============================================================================

class KernelCatalogTest : public ::testing::Test {
  protected:
    void SetUp() override {
        catalog_ = &getKernelCatalog();
    }

    const std::vector<KernelCatalogEntry>* catalog_;
};

TEST_F(KernelCatalogTest, CatalogNotEmpty) {
    EXPECT_GT(catalog_->size(), 0u) << "Catalog should contain at least one kernel";
}

TEST_F(KernelCatalogTest, CatalogHasStandardKernels) {
    int standard_count = 0;
    for (const auto& entry : *catalog_) {
        if (entry.type == KernelType::Standard) {
            standard_count++;
        }
    }
    EXPECT_GE(standard_count, 4)
        << "Should have at least 4 standard kernels (Naive, Tiled, BankConflictFree, DoubleBuffer)";
}

TEST_F(KernelCatalogTest, CatalogHasTensorCoreKernels) {
    int tc_count = 0;
    for (const auto& entry : *catalog_) {
        if (entry.type == KernelType::TensorCore) {
            tc_count++;
        }
    }
    EXPECT_GE(tc_count, 1)
        << "Should have at least 1 tensor core kernel (end-to-end)";
}

TEST_F(KernelCatalogTest, CatalogEntriesHaveNamesAndLaunchers) {
    for (const auto& entry : *catalog_) {
        EXPECT_FALSE(entry.name.empty()) << "All entries should have non-empty names";
        EXPECT_TRUE(static_cast<bool>(entry.launcher))
            << "Entry '" << entry.name << "' should have a valid launcher";
    }
}

TEST_F(KernelCatalogTest, CatalogLaunchCallable) {
    ASSERT_FALSE(catalog_->empty()) << "Catalog is empty, cannot test launch";
    
    // Small test: verify we can call the launcher without crashing
    const auto& entry = (*catalog_)[0];
    const int M = 64, K = 64, N = 64;
    
    DeviceMemory<float> d_A(M * K);
    DeviceMemory<float> d_B(K * N);
    DeviceMemory<float> d_C(M * N);
    
    // Initialize with zeros
    CUDA_CHECK(cudaMemset(d_A.get(), 0, M * K * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_B.get(), 0, K * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_C.get(), 0, M * N * sizeof(float)));
    
    // Launch should not crash
    EXPECT_NO_THROW({
        entry.launcher(d_A.get(), d_B.get(), d_C.get(), M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    });
}

TEST_F(KernelCatalogTest, CatalogPreservesOrder) {
    ASSERT_GE(catalog_->size(), 5u) << "Expected at least 5 kernels";
    
    // Verify the expected order: Naive, Tiled, BankConflictFree, DoubleBuffer, TensorCore end-to-end
    EXPECT_EQ((*catalog_)[0].name, "Naive") << "First kernel should be Naive";
    EXPECT_EQ((*catalog_)[1].name, "Tiled (32x32)") << "Second kernel should be Tiled";
    EXPECT_EQ((*catalog_)[2].name, "Bank Conflict Free") << "Third kernel should be BankConflictFree";
    EXPECT_EQ((*catalog_)[3].name, "Double Buffer") << "Fourth kernel should be DoubleBuffer";
    EXPECT_EQ((*catalog_)[4].name, "Tensor Core (WMMA end-to-end)") << "Fifth kernel should be Tensor Core end-to-end";
}

