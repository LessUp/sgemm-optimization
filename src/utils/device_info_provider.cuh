#pragma once

#include <cuda_runtime.h>

// ============================================================================
// Device Info Provider - Capability Query Seam
// ============================================================================
//
// This seam extracts device capability queries from the singleton pattern,
// allowing tests to inject fake device info without real GPU hardware.
//
// Design:
// - Lightweight struct-based interface (no virtual dispatch overhead)
// - Production adapter backed by DeviceInfoCache singleton
// - Overloaded API: no-arg versions use production adapter by default
// - Tests can supply custom providers for capability scenarios
// ============================================================================

/**
 * Device capability information interface
 *
 * Provides a minimal surface for querying GPU capabilities needed by
 * Tensor Core logic and benchmark calculations.
 */
struct DeviceInfoProvider {
    const cudaDeviceProp *prop;
    int cores_per_sm;
    float clock_ghz;

    /// Check if device supports Tensor Cores (sm_70+)
    bool hasTensorCores() const { return prop->major >= 7; }

    /// Get compute capability major version
    int computeMajor() const { return prop->major; }

    /// Get compute capability minor version
    int computeMinor() const { return prop->minor; }

    /// Get number of streaming multiprocessors
    int smCount() const { return prop->multiProcessorCount; }

    /// Get memory bus width in bits
    int memoryBusWidth() const { return prop->memoryBusWidth; }

    /// Get memory clock rate in kHz
    int memoryClockRate() const { return prop->memoryClockRate; }

    /// Get clock rate in kHz
    int clockRate() const { return prop->clockRate; }
};

// Forward declaration of DeviceInfoCache to avoid circular dependency
class DeviceInfoCache;

/**
 * Production adapter: queries from DeviceInfoCache singleton
 *
 * This is the default provider used by production code paths.
 */
struct ProductionDeviceInfoProvider {
    DeviceInfoProvider get() const;
};

/**
 * Convenience function: get production device info
 *
 * Used as default parameter in overloaded API functions.
 */
inline DeviceInfoProvider getProductionDeviceInfo();

// ============================================================================
// Implementation (inline to avoid linker issues in header-only library)
// ============================================================================

// Implementation moved after DeviceInfoCache definition in cuda_utils.cuh
// to avoid circular dependency. See cuda_utils.cuh for:
//   - ProductionDeviceInfoProvider::get()
//   - getProductionDeviceInfo()
