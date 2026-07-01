/**
 * Kernel catalog module test suite
 *
 * Tests the kernel catalog registration system:
 * - Catalog contains expected kernels
 * - Entries have valid metadata (names, launchers, constraints)
 * - Constraints checking works correctly
 * - Launch functions are callable without crashes
 * - Order preservation of kernel entries
 *
 * This test requires a CUDA device and will be skipped
 * automatically when no CUDA device is available.
 */

#include "gtest_cuda_environment.cuh"
#include "kernels/kernel_catalog.cuh"
#include "utils/benchmark_settings.cuh"
#include "utils/cuda_utils.cuh"
#include <gtest/gtest.h>

// ============================================================================
// Kernel Catalog Metadata Tests
// ============================================================================

class KernelCatalogTest : public ::testing::Test {
  protected:
    void SetUp() override { catalog_ = &getKernelCatalog(); }

    const std::vector<KernelCatalogEntry> *catalog_;
};

TEST_F(KernelCatalogTest, CatalogNotEmpty) {
    EXPECT_GT(catalog_->size(), 0u) << "Catalog should contain at least one kernel";
}

TEST_F(KernelCatalogTest, CatalogHasStandardKernels) {
    int standard_count = countKernelsByType(KernelType::Standard);
    EXPECT_GE(standard_count, 4)
        << "Should have at least 4 standard kernels (Naive, Tiled, BankConflictFree, DoubleBuffer)";
}

TEST_F(KernelCatalogTest, CatalogHasTensorCoreKernels) {
    int tc_count = countKernelsByType(KernelType::TensorCore);
    EXPECT_GE(tc_count, 1) << "Should have at least 1 tensor core kernel (end-to-end)";
}

TEST_F(KernelCatalogTest, CatalogEntriesHaveNamesAndLaunchers) {
    for (const auto &entry : *catalog_) {
        EXPECT_FALSE(entry.name.empty()) << "All entries should have non-empty names";
        EXPECT_TRUE(static_cast<bool>(entry.launcher))
            << "Entry '" << entry.name << "' should have a valid launcher";
    }
}

TEST_F(KernelCatalogTest, CatalogEntriesHaveConstraints) {
    for (const auto &entry : *catalog_) {
        // Standard kernels should not require tensor cores
        if (entry.type == KernelType::Standard) {
            EXPECT_FALSE(entry.constraints.requires_tensor_cores)
                << "Standard kernel '" << entry.name << "' should not require tensor cores";
        }

        // Tensor Core kernels should require sm_70+
        if (entry.type == KernelType::TensorCore) {
            EXPECT_TRUE(entry.constraints.requires_tensor_cores)
                << "Tensor Core kernel '" << entry.name << "' should require tensor cores";
        }
    }
}

TEST_F(KernelCatalogTest, CatalogPreservesOrder) {
    ASSERT_GE(catalog_->size(), 5u) << "Expected at least 5 kernels";

    // Verify the expected order: Naive, Tiled, BankConflictFree, DoubleBuffer, TensorCore
    // end-to-end
    EXPECT_EQ((*catalog_)[0].name, "Naive") << "First kernel should be Naive";
    EXPECT_EQ((*catalog_)[1].name, "Tiled (32x32)") << "Second kernel should be Tiled";
    EXPECT_EQ((*catalog_)[2].name, "Bank Conflict Free")
        << "Third kernel should be BankConflictFree";
    EXPECT_EQ((*catalog_)[3].name, "Double Buffer") << "Fourth kernel should be DoubleBuffer";
    EXPECT_EQ((*catalog_)[4].name, "Tensor Core (WMMA end-to-end)")
        << "Fifth kernel should be Tensor Core end-to-end";
}

// ============================================================================
// Kernel Constraints Tests
// ============================================================================

TEST(KernelConstraintsTest, StandardConstraintsAllowAnyDimensions) {
    auto constraints = KernelConstraints::standard();

    EXPECT_FALSE(constraints.requires_tensor_cores);
    EXPECT_EQ(constraints.dimension_alignment, 0);

    // Should accept any positive dimensions
    EXPECT_TRUE(constraints.isSatisfied(1, 1, 1, false));
    EXPECT_TRUE(constraints.isSatisfied(1024, 1024, 1024, false));
    EXPECT_TRUE(constraints.isSatisfied(15, 17, 33, false)); // Unaligned
}

TEST(KernelConstraintsTest, TensorCoreConstraintsRequireAlignment) {
    auto constraints = KernelConstraints::tensorCore();

    EXPECT_TRUE(constraints.requires_tensor_cores);
    EXPECT_EQ(constraints.dimension_alignment, 16);

    // Should reject without tensor cores
    EXPECT_FALSE(constraints.isSatisfied(16, 16, 16, false));

    // Should accept aligned dimensions with tensor cores
    EXPECT_TRUE(constraints.isSatisfied(16, 16, 16, true));
    EXPECT_TRUE(constraints.isSatisfied(64, 128, 256, true));

    // Should reject unaligned dimensions even with tensor cores
    EXPECT_FALSE(constraints.isSatisfied(15, 16, 16, true));
    EXPECT_FALSE(constraints.isSatisfied(16, 17, 16, true));
    EXPECT_FALSE(constraints.isSatisfied(16, 16, 33, true));
}

TEST(KernelConstraintsTest, ComputeOnlyConstraintsAreSpecial) {
    auto constraints = KernelConstraints::tensorCoreComputeOnly();

    EXPECT_TRUE(constraints.requires_tensor_cores);
    EXPECT_TRUE(constraints.requires_compute_only);
    EXPECT_EQ(constraints.dimension_alignment, 16);
}

// ============================================================================
// Catalog Entry Tests
// ============================================================================

TEST(KernelCatalogEntryTest, DefaultToleranceForStandardKernels) {
    auto constraints = KernelConstraints::standard();
    KernelCatalogEntry entry{"Test Standard", KernelType::Standard, nullptr, constraints};

    auto tolerance = entry.defaultTolerance();
    EXPECT_FLOAT_EQ(tolerance.rtol, kStandardVerifyTolerance.rtol);
    EXPECT_FLOAT_EQ(tolerance.atol, kStandardVerifyTolerance.atol);
}

TEST(KernelCatalogEntryTest, DefaultToleranceForTensorCoreKernels) {
    auto constraints = KernelConstraints::tensorCore();
    KernelCatalogEntry entry{"Test TensorCore", KernelType::TensorCore, nullptr, constraints};

    auto tolerance = entry.defaultTolerance();
    EXPECT_FLOAT_EQ(tolerance.rtol, kTensorCoreVerifyTolerance.rtol);
    EXPECT_FLOAT_EQ(tolerance.atol, kTensorCoreVerifyTolerance.atol);
}

TEST(KernelCatalogEntryTest, CanRunMethodWorks) {
    auto tc_entry = getTensorCoreComputeOnlyEntry();

    // Should not run without tensor cores
    EXPECT_FALSE(tc_entry.canRun(16, 16, 16, false));

    // Should run with tensor cores and aligned dimensions
    EXPECT_TRUE(tc_entry.canRun(16, 16, 16, true));
    EXPECT_TRUE(tc_entry.canRun(64, 128, 256, true));

    // Should not run with unaligned dimensions
    EXPECT_FALSE(tc_entry.canRun(15, 16, 16, true));
}

// ============================================================================
// Catalog Query Utilities Tests
// ============================================================================

TEST(CatalogQueryTest, CountKernelsByTypeWorks) {
    int standard_count = countKernelsByType(KernelType::Standard);
    int tensor_core_count = countKernelsByType(KernelType::TensorCore);

    EXPECT_GE(standard_count, 4);
    EXPECT_GE(tensor_core_count, 1);
}

TEST(CatalogQueryTest, GetKernelNamesWorks) {
    auto standard_names = getKernelNames(KernelType::Standard);

    EXPECT_GE(standard_names.size(), 4u);
    EXPECT_EQ(standard_names[0], "Naive");
    EXPECT_EQ(standard_names[1], "Tiled (32x32)");
}

TEST(CatalogQueryTest, CanRunTensorCoreKernelsWorks) {
    // With tensor cores
    EXPECT_TRUE(canRunTensorCoreKernels(16, 16, 16, true));
    EXPECT_TRUE(canRunTensorCoreKernels(64, 128, 256, true));
    EXPECT_FALSE(canRunTensorCoreKernels(15, 16, 16, true)); // Unaligned

    // Without tensor cores
    EXPECT_FALSE(canRunTensorCoreKernels(16, 16, 16, false));
}

// ============================================================================
// Kernel Launch Tests (requires CUDA device)
// ============================================================================

TEST_F(KernelCatalogTest, CatalogLaunchCallable) {
    ASSERT_FALSE(catalog_->empty()) << "Catalog is empty, cannot test launch";

    // Small test: verify we can call the launcher without crashing
    const auto &entry = (*catalog_)[0];
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

int main(int argc, char **argv) { return runCudaAwareTests(argc, argv); }
