/**
 * Benchmark Settings Module Test Suite
 *
 * Tests for the centralized benchmark settings module that consolidates:
 * - Run configuration (warmup, benchmark iterations)
 * - Verification tolerance policy
 * - Output/export options
 */

#include <gtest/gtest.h>
#include "utils/benchmark_settings.cuh"

// ============================================================================
// Run Settings Tests
// ============================================================================

TEST(RunSettingsTest, DefaultValues) {
    RunSettings settings;
    EXPECT_EQ(settings.warmup_runs, 5);
    EXPECT_EQ(settings.benchmark_runs, 20);
}

TEST(RunSettingsTest, CustomValues) {
    RunSettings settings{10, 50};
    EXPECT_EQ(settings.warmup_runs, 10);
    EXPECT_EQ(settings.benchmark_runs, 50);
}

TEST(RunSettingsTest, ZeroWarmupIsValid) {
    RunSettings settings{0, 20};
    EXPECT_EQ(settings.warmup_runs, 0);
    EXPECT_EQ(settings.benchmark_runs, 20);
}

// ============================================================================
// Verification Settings Tests
// ============================================================================

TEST(VerificationSettingsTest, DefaultIsStandardTolerance) {
    VerificationSettings settings;
    EXPECT_FLOAT_EQ(settings.tolerance.rtol, kStandardVerifyTolerance.rtol);
    EXPECT_FLOAT_EQ(settings.tolerance.atol, kStandardVerifyTolerance.atol);
}

TEST(VerificationSettingsTest, TensorCoreToleranceOption) {
    VerificationSettings settings = VerificationSettings::tensorCore();
    EXPECT_FLOAT_EQ(settings.tolerance.rtol, kTensorCoreVerifyTolerance.rtol);
    EXPECT_FLOAT_EQ(settings.tolerance.atol, kTensorCoreVerifyTolerance.atol);
}

TEST(VerificationSettingsTest, CustomTolerance) {
    VerifyTolerance custom{0.01f, 0.001f};
    VerificationSettings settings{custom};
    EXPECT_FLOAT_EQ(settings.tolerance.rtol, 0.01f);
    EXPECT_FLOAT_EQ(settings.tolerance.atol, 0.001f);
}

// ============================================================================
// Output Settings Tests
// ============================================================================

TEST(OutputSettingsTest, DefaultEnablesRooflineExport) {
    OutputSettings settings;
    EXPECT_TRUE(settings.export_roofline);
}

TEST(OutputSettingsTest, CanDisableRooflineExport) {
    OutputSettings settings{false};
    EXPECT_FALSE(settings.export_roofline);
}

TEST(OutputSettingsTest, DefaultFilenameGeneration) {
    OutputSettings settings;
    std::string filename = settings.makeRooflineFilename(1024, 1024, 1024);
    EXPECT_EQ(filename, "roofline_data_1024_1024_1024.csv");
}

TEST(OutputSettingsTest, CustomFilenamePattern) {
    OutputSettings settings;
    settings.filename_pattern = "bench_{M}x{K}x{N}.csv";
    std::string filename = settings.makeRooflineFilename(512, 768, 1024);
    EXPECT_EQ(filename, "bench_512x768x1024.csv");
}

// ============================================================================
// Benchmark Settings Tests
// ============================================================================

TEST(BenchmarkSettingsTest, DefaultSettings) {
    BenchmarkSettings settings;
    EXPECT_EQ(settings.run.warmup_runs, 5);
    EXPECT_EQ(settings.run.benchmark_runs, 20);
    EXPECT_FLOAT_EQ(settings.verify.tolerance.rtol, kStandardVerifyTolerance.rtol);
    EXPECT_TRUE(settings.output.export_roofline);
}

TEST(BenchmarkSettingsTest, CustomRunSettings) {
    BenchmarkSettings settings;
    settings.run.warmup_runs = 10;
    settings.run.benchmark_runs = 50;
    EXPECT_EQ(settings.run.warmup_runs, 10);
    EXPECT_EQ(settings.run.benchmark_runs, 50);
}

TEST(BenchmarkSettingsTest, CustomVerificationSettings) {
    BenchmarkSettings settings;
    settings.verify = VerificationSettings::tensorCore();
    EXPECT_FLOAT_EQ(settings.verify.tolerance.rtol, kTensorCoreVerifyTolerance.rtol);
}

TEST(BenchmarkSettingsTest, CustomOutputSettings) {
    BenchmarkSettings settings;
    settings.output.export_roofline = false;
    EXPECT_FALSE(settings.output.export_roofline);
}

TEST(BenchmarkSettingsTest, ToleranceForKernelType) {
    BenchmarkSettings settings;
    
    // Standard kernels use standard tolerance
    VerifyTolerance std_tol = settings.toleranceForKernel(KernelType::Standard);
    EXPECT_FLOAT_EQ(std_tol.rtol, kStandardVerifyTolerance.rtol);
    
    // Tensor Core kernels use tensor core tolerance
    VerifyTolerance tc_tol = settings.toleranceForKernel(KernelType::TensorCore);
    EXPECT_FLOAT_EQ(tc_tol.rtol, kTensorCoreVerifyTolerance.rtol);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
