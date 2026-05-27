# Device Capability Seam

## Overview

The device capability seam extracts GPU capability queries from the singleton pattern (`DeviceInfoCache::instance()`), allowing tests to inject fake device info without requiring specific GPU hardware.

## Design

### Core Interface

`DeviceInfoProvider` (in `src/utils/device_info_provider.cuh`):
- Lightweight struct-based interface (no virtual dispatch overhead)
- Provides minimal surface for querying GPU capabilities
- Methods: `hasTensorCores()`, `computeMajor()`, `computeMinor()`, `smCount()`, etc.

### Production Adapter

`ProductionDeviceInfoProvider`:
- Backed by `DeviceInfoCache` singleton
- Used by default in all production code paths
- Zero runtime overhead compared to direct singleton access

### API Pattern

All capability-dependent functions provide two overloads:

```cpp
// With explicit provider (for tests)
bool tensorCoresAvailable(const DeviceInfoProvider &provider);
float getTheoreticalPeakGflops(const DeviceInfoProvider &provider);

// Default (uses production provider)
bool tensorCoresAvailable();
float getTheoreticalPeakGflops();
```

## Usage

### Production Code

Production code continues to use no-argument versions:

```cpp
if (tensorCoresAvailable()) {
    // Use Tensor Cores
}
float peak = getTheoreticalPeakGflops();
```

### Test Code

Tests can inject fake device info:

```cpp
cudaDeviceProp fake_prop{};
fake_prop.major = 7;  // Volta
fake_prop.minor = 0;
fake_prop.multiProcessorCount = 80;

DeviceInfoProvider fake_provider{&fake_prop, 64, 1.53f};

EXPECT_TRUE(tensorCoresAvailable(fake_provider));
EXPECT_NEAR(getTheoreticalPeakGflops(fake_provider), 15667.2f, 0.1f);
```

## Affected Modules

### `src/utils/cuda_utils.cuh`
- Added production adapter implementation
- `DeviceInfoCache` remains unchanged (still singleton)
- Only the production adapter calls the singleton

### `src/kernels/tensor_core_sgemm.cuh`
- `tensorCoresAvailable()` - overloaded
- `getTensorCoreArchName()` - overloaded
- No longer calls `DeviceInfoCache::instance()` directly

### `src/utils/benchmark_metrics.cuh`
- `getTheoreticalPeakGflops()` - overloaded
- `getTheoreticalPeakBandwidth()` - overloaded
- No longer calls `DeviceInfoCache::instance()` directly

## Testing

See `tests/test_device_info_seam.cu` for comprehensive examples:

- Fake device provider creation
- Tensor Core capability testing without real hardware
- Peak GFLOPS/bandwidth calculations with fake devices
- Architectural classification (Volta, Ampere, Hopper, etc.)
- Production adapter integration tests

## Design Rationale

### Why struct-based instead of virtual interfaces?

- Zero runtime overhead (no vtable lookups)
- Header-only implementation remains simple
- Sufficient for the single use case (production vs test)

### Why preserve the singleton?

- Minimal code changes to existing production paths
- The singleton itself isn't problematic; the hard dependency was
- Production adapter provides the indirection we need

### Why overloads instead of default parameters?

- Clearer call sites in tests
- No ambiguity about which version is being called
- Easier to grep for test-specific vs production usage

## Limitations

- Tests still require a CUDA-capable GPU to run (for initialization)
- Only capability queries are abstracted, not actual kernel execution
- The seam is focused on decision logic, not full GPU emulation

## Future Extensions

If needed, this seam could be extended to support:
- Mock kernel execution for unit tests
- Multiple device scenarios in a single test
- Device capability fuzzing for edge case discovery
