/**
 * Device Info Provider CUDA Tests
 *
 * Tests that require a real CUDA device. These tests will be skipped
 * automatically when no CUDA device is available.
 */

#include <gtest/gtest.h>

#include "gtest_cuda_environment.cuh"
#include "utils/benchmark_metrics.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/device_info_provider.cuh"

namespace {

// ============================================================================
// Production Adapter Integration Test
// ============================================================================

TEST(DeviceInfoSeam, ProductionAdapterWorks) {
    // This test validates that the production adapter can successfully query
    // real device info. It should pass on any CUDA-capable device.
    DeviceInfoProvider prod = getProductionDeviceInfo();

    // Basic sanity checks
    EXPECT_NE(prod.prop, nullptr);
    EXPECT_GT(prod.cores_per_sm, 0);
    EXPECT_GT(prod.clock_ghz, 0.0f);

    // Check that overloaded functions work without provider parameter
    float peak_gflops = getTheoreticalPeakGflops();
    EXPECT_GT(peak_gflops, 0.0f);

    float peak_bandwidth = getTheoreticalPeakBandwidth();
    EXPECT_GT(peak_bandwidth, 0.0f);

    // Tensor core availability should be consistent
    bool has_tc_explicit = tensorCoresAvailable(prod);
    bool has_tc_default = tensorCoresAvailable();
    EXPECT_EQ(has_tc_explicit, has_tc_default);
}

} // namespace

int main(int argc, char **argv) { return runCudaAwareTests(argc, argv); }
