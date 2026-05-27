/**
 * Device Info Provider Seam Tests
 *
 * Tests for the device capability query seam, demonstrating that tests can
 * inject fake device info without relying on real GPU hardware capabilities.
 */

#include <gtest/gtest.h>

#include "kernels/tensor_core_sgemm.cuh"
#include "utils/benchmark_metrics.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/device_info_provider.cuh"

namespace {

/**
 * Test fixture with fake device properties
 */
class FakeDeviceProvider : public ::testing::Test {
  protected:
    void SetUp() override {
        // Initialize a fake Volta (sm_70) device
        memset(&volta_prop_, 0, sizeof(cudaDeviceProp));
        volta_prop_.major = 7;
        volta_prop_.minor = 0;
        volta_prop_.multiProcessorCount = 80;
        volta_prop_.clockRate = 1530000; // 1.53 GHz in kHz
        volta_prop_.memoryClockRate = 877000; // 877 MHz in kHz
        volta_prop_.memoryBusWidth = 4096; // HBM2

        volta_provider_ = DeviceInfoProvider{
            &volta_prop_,
            64,    // Volta cores per SM
            1.53f, // Clock in GHz
        };

        // Initialize a fake pre-Volta (sm_60) device
        memset(&pre_volta_prop_, 0, sizeof(cudaDeviceProp));
        pre_volta_prop_.major = 6;
        pre_volta_prop_.minor = 1;
        pre_volta_prop_.multiProcessorCount = 20;
        pre_volta_prop_.clockRate = 1733000;
        pre_volta_prop_.memoryClockRate = 4513000;
        pre_volta_prop_.memoryBusWidth = 256;

        pre_volta_provider_ = DeviceInfoProvider{
            &pre_volta_prop_,
            128, // Pascal cores per SM
            1.733f,
        };

        // Initialize a fake Ampere (sm_80) device
        memset(&ampere_prop_, 0, sizeof(cudaDeviceProp));
        ampere_prop_.major = 8;
        ampere_prop_.minor = 0;
        ampere_prop_.multiProcessorCount = 108;
        ampere_prop_.clockRate = 1410000;
        ampere_prop_.memoryClockRate = 1593000;
        ampere_prop_.memoryBusWidth = 5120;

        ampere_provider_ = DeviceInfoProvider{
            &ampere_prop_,
            64, // A100 cores per SM
            1.41f,
        };
    }

    cudaDeviceProp volta_prop_;
    DeviceInfoProvider volta_provider_;

    cudaDeviceProp pre_volta_prop_;
    DeviceInfoProvider pre_volta_provider_;

    cudaDeviceProp ampere_prop_;
    DeviceInfoProvider ampere_provider_;
};

// ============================================================================
// Tensor Core Capability Tests
// ============================================================================

TEST_F(FakeDeviceProvider, VoltaHasTensorCores) {
    EXPECT_TRUE(tensorCoresAvailable(volta_provider_));
    EXPECT_STREQ(getTensorCoreArchName(volta_provider_), "Volta");
}

TEST_F(FakeDeviceProvider, PreVoltaNoTensorCores) {
    EXPECT_FALSE(tensorCoresAvailable(pre_volta_provider_));
}

TEST_F(FakeDeviceProvider, AmpereHasTensorCores) {
    EXPECT_TRUE(tensorCoresAvailable(ampere_provider_));
    EXPECT_STREQ(getTensorCoreArchName(ampere_provider_), "Ampere");
}

// ============================================================================
// Benchmark Metrics Tests
// ============================================================================

TEST_F(FakeDeviceProvider, VoltaPeakGflopsCalculation) {
    // Volta V100: 80 SMs * 64 cores/SM * 2 (FMA) * 1.53 GHz * 1000
    // Expected: ~15.667 TFLOPS
    float peak_gflops = getTheoreticalPeakGflops(volta_provider_);
    EXPECT_NEAR(peak_gflops, 15667.2f, 0.1f);
}

TEST_F(FakeDeviceProvider, AmperePeakGflopsCalculation) {
    // A100: 108 SMs * 64 cores/SM * 2 (FMA) * 1.41 GHz * 1000
    // Expected: ~19.481 TFLOPS
    float peak_gflops = getTheoreticalPeakGflops(ampere_provider_);
    EXPECT_NEAR(peak_gflops, 19481.0f, 0.1f);
}

TEST_F(FakeDeviceProvider, VoltaPeakBandwidthCalculation) {
    // Volta V100: 2 (DDR) * 877 MHz * (4096 bits / 8 bytes/bit) / 1000
    // Expected: ~900 GB/s
    float peak_bandwidth = getTheoreticalPeakBandwidth(volta_provider_);
    EXPECT_NEAR(peak_bandwidth, 898.048f, 0.1f);
}

TEST_F(FakeDeviceProvider, PreVoltaPeakBandwidthCalculation) {
    // Pascal GP102: 2 * 4513 MHz * (256 / 8) / 1000
    // Expected: ~288 GB/s
    float peak_bandwidth = getTheoreticalPeakBandwidth(pre_volta_provider_);
    EXPECT_NEAR(peak_bandwidth, 288.832f, 0.1f);
}

// ============================================================================
// Architectural Classification Tests
// ============================================================================

TEST_F(FakeDeviceProvider, ArchitectureNamingVolta) {
    EXPECT_STREQ(getTensorCoreArchName(volta_provider_), "Volta");
}

TEST_F(FakeDeviceProvider, ArchitectureNamingAmpere) {
    EXPECT_STREQ(getTensorCoreArchName(ampere_provider_), "Ampere");
}

TEST_F(FakeDeviceProvider, ArchitectureNamingTuring) {
    cudaDeviceProp turing_prop{};
    turing_prop.major = 7;
    turing_prop.minor = 5;
    DeviceInfoProvider turing_provider{&turing_prop, 64, 1.5f};

    EXPECT_STREQ(getTensorCoreArchName(turing_provider), "Turing");
}

TEST_F(FakeDeviceProvider, ArchitectureNamingHopper) {
    cudaDeviceProp hopper_prop{};
    hopper_prop.major = 9;
    hopper_prop.minor = 0;
    DeviceInfoProvider hopper_provider{&hopper_prop, 128, 1.8f};

    EXPECT_STREQ(getTensorCoreArchName(hopper_provider), "Hopper");
}

TEST_F(FakeDeviceProvider, ArchitectureNamingUnknown) {
    cudaDeviceProp unknown_prop{};
    unknown_prop.major = 5;
    unknown_prop.minor = 2;
    DeviceInfoProvider unknown_provider{&unknown_prop, 128, 1.0f};

    EXPECT_STREQ(getTensorCoreArchName(unknown_provider), "Unknown");
}

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

// ============================================================================
// Dimension Support Tests (no device dependency)
// ============================================================================

TEST(TensorCoreDimensions, AlignedDimensionsSupported) {
    EXPECT_TRUE(tensorCoreDimensionsSupported(16, 16, 16));
    EXPECT_TRUE(tensorCoreDimensionsSupported(32, 32, 32));
    EXPECT_TRUE(tensorCoreDimensionsSupported(64, 128, 256));
    EXPECT_TRUE(tensorCoreDimensionsSupported(256, 256, 256));
}

TEST(TensorCoreDimensions, UnalignedDimensionsNotSupported) {
    EXPECT_FALSE(tensorCoreDimensionsSupported(15, 16, 16));
    EXPECT_FALSE(tensorCoreDimensionsSupported(16, 17, 16));
    EXPECT_FALSE(tensorCoreDimensionsSupported(16, 16, 33));
    EXPECT_FALSE(tensorCoreDimensionsSupported(511, 513, 1025));
}

TEST(TensorCoreDimensions, ZeroDimensionsNotSupported) {
    EXPECT_FALSE(tensorCoreDimensionsSupported(0, 16, 16));
    EXPECT_FALSE(tensorCoreDimensionsSupported(16, 0, 16));
    EXPECT_FALSE(tensorCoreDimensionsSupported(16, 16, 0));
}

TEST(TensorCoreDimensions, NegativeDimensionsNotSupported) {
    EXPECT_FALSE(tensorCoreDimensionsSupported(-16, 16, 16));
    EXPECT_FALSE(tensorCoreDimensionsSupported(16, -16, 16));
    EXPECT_FALSE(tensorCoreDimensionsSupported(16, 16, -16));
}

} // namespace
