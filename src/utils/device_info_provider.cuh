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
 *
 * Contract:
 * - `prop` must never be null; all accessors dereference without null checks
 * - `prop` must remain valid for the lifetime of this provider instance
 * - `cores_per_sm` and `clock_ghz` are precomputed adapter values carried for
 *   testability and metric calculations; they are NOT directly queryable from
 *   cudaDeviceProp and must be supplied by the provider (e.g., via
 *   DeviceInfoCache architecture tables or test fixtures)
 */
struct DeviceInfoProvider {
    const cudaDeviceProp *prop;
    int cores_per_sm;
    float clock_ghz;

    /// Check if device supports Tensor Cores (sm_70+)
    /// @pre prop != nullptr
    bool hasTensorCores() const { return prop->major >= 7; }

    /// Get compute capability major version
    /// @pre prop != nullptr
    int computeMajor() const { return prop->major; }

    /// Get compute capability minor version
    /// @pre prop != nullptr
    int computeMinor() const { return prop->minor; }

    /// Get number of streaming multiprocessors
    /// @pre prop != nullptr
    int smCount() const { return prop->multiProcessorCount; }

    /// Get memory bus width in bits
    /// @pre prop != nullptr
    int memoryBusWidth() const { return prop->memoryBusWidth; }

    /// Get memory clock rate in kHz
    /// @pre prop != nullptr
    int memoryClockRate() const { return prop->memoryClockRate; }

    /// Get clock rate in kHz
    /// @pre prop != nullptr
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
