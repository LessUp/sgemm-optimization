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
    VerifyTolerance standard_tolerance = kStandardVerifyTolerance;
    VerifyTolerance tensor_core_tolerance = kTensorCoreVerifyTolerance;

    static VerificationSettings defaults() {
        return VerificationSettings{};
    }
};

// ============================================================================
// Output Settings
// ============================================================================

struct OutputSettings {
    bool export_roofline = true;
    std::string filename_pattern = "roofline_data_{M}_{K}_{N}.csv";

    std::string makeRooflineFilename(int M, int K, int N) const {
        std::string result = filename_pattern;
        
        // Replace all occurrences of each token
        auto replaceAll = [](std::string& str, const std::string& from, const std::string& to) {
            size_t pos = 0;
            while ((pos = str.find(from, pos)) != std::string::npos) {
                str.replace(pos, from.length(), to);
                pos += to.length();
            }
        };
        
        replaceAll(result, "{M}", std::to_string(M));
        replaceAll(result, "{K}", std::to_string(K));
        replaceAll(result, "{N}", std::to_string(N));
        
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
            return verify.standard_tolerance;
        case KernelType::TensorCore:
            return verify.tensor_core_tolerance;
        default:
            return verify.standard_tolerance;
        }
    }
};
