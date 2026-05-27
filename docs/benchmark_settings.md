# Benchmark Settings Module

## Overview

The benchmark settings module centralizes configuration for benchmark runs, verification tolerances, and output options. This eliminates scattered magic constants and hardcoded policies across CLI parsing, benchmark orchestration, and verification.

## Components

### RunSettings
Controls warmup and benchmark iteration counts.

```cpp
RunSettings run;
run.warmup_runs = 10;      // Default: 5
run.benchmark_runs = 50;   // Default: 20
```

### VerificationSettings
Manages tolerance policy for kernel correctness verification.

```cpp
VerificationSettings verify;

// Standard FP32 tolerance (default)
verify = VerificationSettings::standard();  // rtol=1e-3, atol=1e-4

// Relaxed Tensor Core tolerance for mixed-precision
verify = VerificationSettings::tensorCore(); // rtol=5e-2, atol=1e-2

// Custom tolerance
verify.tolerance = VerifyTolerance{0.01f, 0.001f};
```

### OutputSettings
Controls roofline data export behavior and filename generation.

```cpp
OutputSettings output;
output.export_roofline = true;  // Default: true
output.filename_pattern = "roofline_data_{M}_{K}_{N}.csv";  // Default

// Generate filename for specific dimensions
std::string filename = output.makeRooflineFilename(1024, 1024, 1024);
// Result: "roofline_data_1024_1024_1024.csv"
```

### BenchmarkSettings (Aggregate)
Combines all settings with kernel-type-aware tolerance selection.

```cpp
BenchmarkSettings settings;

// Access components
settings.run.warmup_runs = 10;
settings.verify = VerificationSettings::tensorCore();
settings.output.export_roofline = false;

// Helper: select tolerance by kernel type
VerifyTolerance tol = settings.toleranceForKernel(KernelType::Standard);
```

## Builder Pattern (Optional)

For fluent configuration:

```cpp
BenchmarkSettings settings = BenchmarkSettings::builder()
    .withWarmupRuns(10)
    .withBenchmarkRuns(50)
    .withTensorCoreTolerance()
    .withoutRooflineExport()
    .build();
```

## Integration

### CLI Parser
`BenchmarkConfig` now embeds `BenchmarkSettings`:

```cpp
BenchmarkConfig config;
CliParser parser(argc, argv);
parser.parse(config);

// Settings are built by CLI flags:
// --warmup N       -> config.settings.run.warmup_runs
// --benchmark N    -> config.settings.run.benchmark_runs
```

### Benchmark Runner
Consumes settings instead of hardcoding policies:

```cpp
BenchmarkRunner runner(config);

// Uses config.settings.run.{warmup_runs, benchmark_runs}
// Uses config.settings.toleranceForKernel(type)
// Exports roofline data only if config.settings.output.export_roofline
```

## Benefits

1. **Locality**: Run counts, tolerances, and export policy in one place
2. **Testability**: Settings can be constructed independently of CLI or orchestration
3. **Extensibility**: Easy to add new settings (e.g., output directory, format options)
4. **No magic constants**: Hardcoded tolerances (`kStandardVerifyTolerance`) now selected explicitly via `toleranceForKernel()`
5. **Conditional export**: Roofline export no longer unconditional

## Migration Notes

### Before
```cpp
// CLI parser
config.warmup_runs = 10;
config.benchmark_runs = 50;

// Benchmark runner
benchmark.run(..., config.warmup_runs, config.benchmark_runs, kStandardVerifyTolerance);
benchmark.exportRooflineData("roofline_data_1024_1024_1024.csv");  // Always
```

### After
```cpp
// CLI parser
config.settings.run.warmup_runs = 10;
config.settings.run.benchmark_runs = 50;

// Benchmark runner
VerifyTolerance tol = config.settings.toleranceForKernel(KernelType::Standard);
benchmark.run(..., config.settings.run.warmup_runs, config.settings.run.benchmark_runs, tol);

if (config.settings.output.export_roofline) {
    std::string filename = config.settings.output.makeRooflineFilename(M, K, N);
    benchmark.exportRooflineData(filename);
}
```

## Testing

See `tests/test_benchmark_settings.cu` for comprehensive unit tests covering:
- Default values
- Custom values
- Tolerance selection by kernel type
- Filename generation
- Builder pattern
