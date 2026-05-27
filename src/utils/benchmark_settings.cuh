#pragma once

#include "verify.cuh"
#include <cstdio>
#include <string>

// ============================================================================
// Benchmark Settings Module
//
// Centralizes warmup runs, benchmark runs, verification tolerance policy,
// and output/export options. Eliminates magic constants and hardcoded
// policies scattered across CLI parser, benchmark runner, and verification.
// ============================================================================

// ============================================================================
// Kernel Type Classification
// ============================================================================

enum class KernelType {
    Standard,    // FP32 standard kernels
    TensorCore   // Mixed-precision tensor core kernels
};

// ============================================================================
// Run Settings
// ============================================================================

struct RunSettings {
    int warmup_runs = 5;
    int benchmark_runs = 20;
};

// ============================================================================
// Verification Settings
// ============================================================================

struct VerificationSettings {
    VerifyTolerance tolerance = kStandardVerifyTolerance;

    static VerificationSettings standard() {
        return VerificationSettings{kStandardVerifyTolerance};
    }

    static VerificationSettings tensorCore() {
        return VerificationSettings{kTensorCoreVerifyTolerance};
    }
};

// ============================================================================
// Output Settings
// ============================================================================

struct OutputSettings {
    bool export_roofline = true;
    std::string filename_pattern = "roofline_data_{M}_{K}_{N}.csv";

    std::string makeRooflineFilename(int M, int K, int N) const {
        // Simple pattern replacement
        std::string result = filename_pattern;
        
        // Replace {M}
        size_t pos = result.find("{M}");
        if (pos != std::string::npos) {
            result.replace(pos, 3, std::to_string(M));
        }
        
        // Replace {K}
        pos = result.find("{K}");
        if (pos != std::string::npos) {
            result.replace(pos, 3, std::to_string(K));
        }
        
        // Replace {N}
        pos = result.find("{N}");
        if (pos != std::string::npos) {
            result.replace(pos, 3, std::to_string(N));
        }
        
        return result;
    }
};

// ============================================================================
// Benchmark Settings (Aggregate)
// ============================================================================

struct BenchmarkSettings {
    RunSettings run;
    VerificationSettings verify;
    OutputSettings output;

    // Helper: select tolerance based on kernel type
    VerifyTolerance toleranceForKernel(KernelType type) const {
        switch (type) {
        case KernelType::Standard:
            return kStandardVerifyTolerance;
        case KernelType::TensorCore:
            return kTensorCoreVerifyTolerance;
        default:
            return kStandardVerifyTolerance;
        }
    }
};
