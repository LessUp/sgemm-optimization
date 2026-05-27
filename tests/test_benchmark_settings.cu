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

TEST(VerificationSettingsTest, DefaultTolerances) {
    VerificationSettings settings;
    EXPECT_FLOAT_EQ(settings.standard_tolerance.rtol, kStandardVerifyTolerance.rtol);
    EXPECT_FLOAT_EQ(settings.standard_tolerance.atol, kStandardVerifyTolerance.atol);
    EXPECT_FLOAT_EQ(settings.tensor_core_tolerance.rtol, kTensorCoreVerifyTolerance.rtol);
    EXPECT_FLOAT_EQ(settings.tensor_core_tolerance.atol, kTensorCoreVerifyTolerance.atol);
}

TEST(VerificationSettingsTest, CustomStandardTolerance) {
    VerificationSettings settings;
    settings.standard_tolerance = {0.01f, 0.001f};
    EXPECT_FLOAT_EQ(settings.standard_tolerance.rtol, 0.01f);
    EXPECT_FLOAT_EQ(settings.standard_tolerance.atol, 0.001f);
    // Tensor core tolerance should remain unchanged
    EXPECT_FLOAT_EQ(settings.tensor_core_tolerance.rtol, kTensorCoreVerifyTolerance.rtol);
}

TEST(VerificationSettingsTest, CustomTensorCoreTolerance) {
    VerificationSettings settings;
    settings.tensor_core_tolerance = {0.1f, 0.05f};
    EXPECT_FLOAT_EQ(settings.tensor_core_tolerance.rtol, 0.1f);
    EXPECT_FLOAT_EQ(settings.tensor_core_tolerance.atol, 0.05f);
    // Standard tolerance should remain unchanged
    EXPECT_FLOAT_EQ(settings.standard_tolerance.rtol, kStandardVerifyTolerance.rtol);
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

TEST(OutputSettingsTest, DuplicateTokenReplacement) {
    OutputSettings settings;
    settings.filename_pattern = "{M}x{K}x{N}_again_{M}_{K}_{N}.csv";
    std::string filename = settings.makeRooflineFilename(256, 384, 512);
    EXPECT_EQ(filename, "256x384x512_again_256_384_512.csv");
}

TEST(OutputSettingsTest, ArbitraryTokenOrder) {
    OutputSettings settings;
    settings.filename_pattern = "{N}_{K}_{M}_data.csv";
    std::string filename = settings.makeRooflineFilename(128, 256, 512);
    EXPECT_EQ(filename, "512_256_128_data.csv");
}

// ============================================================================
// Benchmark Settings Tests
// ============================================================================

TEST(BenchmarkSettingsTest, DefaultSettings) {
    BenchmarkSettings settings;
    EXPECT_EQ(settings.run.warmup_runs, 5);
    EXPECT_EQ(settings.run.benchmark_runs, 20);
    EXPECT_FLOAT_EQ(settings.verify.standard_tolerance.rtol, kStandardVerifyTolerance.rtol);
    EXPECT_FLOAT_EQ(settings.verify.tensor_core_tolerance.rtol, kTensorCoreVerifyTolerance.rtol);
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
    settings.verify.standard_tolerance = {0.01f, 0.001f};
    EXPECT_FLOAT_EQ(settings.verify.standard_tolerance.rtol, 0.01f);
}

TEST(BenchmarkSettingsTest, CustomOutputSettings) {
    BenchmarkSettings settings;
    settings.output.export_roofline = false;
    EXPECT_FALSE(settings.output.export_roofline);
}

TEST(BenchmarkSettingsTest, ToleranceForKernelTypeUsesDefaults) {
    BenchmarkSettings settings;
    
    // Standard kernels use standard tolerance
    VerifyTolerance std_tol = settings.toleranceForKernel(KernelType::Standard);
    EXPECT_FLOAT_EQ(std_tol.rtol, kStandardVerifyTolerance.rtol);
    EXPECT_FLOAT_EQ(std_tol.atol, kStandardVerifyTolerance.atol);
    
    // Tensor Core kernels use tensor core tolerance
    VerifyTolerance tc_tol = settings.toleranceForKernel(KernelType::TensorCore);
    EXPECT_FLOAT_EQ(tc_tol.rtol, kTensorCoreVerifyTolerance.rtol);
    EXPECT_FLOAT_EQ(tc_tol.atol, kTensorCoreVerifyTolerance.atol);
}

TEST(BenchmarkSettingsTest, ToleranceForKernelTypeRespectsCustomSettings) {
    BenchmarkSettings settings;
    
    // Customize both tolerances
    settings.verify.standard_tolerance = {0.01f, 0.001f};
    settings.verify.tensor_core_tolerance = {0.1f, 0.05f};
    
    // Verify toleranceForKernel returns the custom values
    VerifyTolerance std_tol = settings.toleranceForKernel(KernelType::Standard);
    EXPECT_FLOAT_EQ(std_tol.rtol, 0.01f);
    EXPECT_FLOAT_EQ(std_tol.atol, 0.001f);
    
    VerifyTolerance tc_tol = settings.toleranceForKernel(KernelType::TensorCore);
    EXPECT_FLOAT_EQ(tc_tol.rtol, 0.1f);
    EXPECT_FLOAT_EQ(tc_tol.atol, 0.05f);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
