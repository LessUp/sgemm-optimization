#pragma once

#include "../utils/benchmark_settings.cuh"
#include "double_buffer_sgemm.cuh"
#include "naive_sgemm.cuh"
#include "tensor_core_fallback.cuh"
#include "tensor_core_sgemm.cuh"
#include "tiled_sgemm.cuh"

#include <functional>
#include <string>
#include <vector>

// ============================================================================
// Kernel Catalog Module
//
// THE authoritative source for kernel ladder metadata.
//
// Design:
// - Single source of truth for kernel name, type, tolerance, launcher, constraints
// - Benchmark, tests, and documentation all reference this catalog
// - New kernels only require adding one entry here
//
// Catalog entries provide:
// - Identity: name, type classification
// - Behavior: launch adapter, default tolerance
// - Constraints: dimension requirements (e.g., Tensor Core requires 16-aligned)
// ============================================================================

// ============================================================================
// Kernel Launch Adapter Type
// ============================================================================

using KernelLauncher = std::function<void(const float *, const float *, float *, int, int, int)>;

// ============================================================================
// Kernel Constraints
// ============================================================================

/**
 * Describes runtime constraints for a kernel.
 *
 * Used by BenchmarkRunner to decide whether a kernel can run with given dimensions.
 */
struct KernelConstraints {
    bool requires_tensor_cores; // Requires sm_70+
    int dimension_alignment;    // All dimensions must be multiple of this (0 = no constraint)
    bool requires_compute_only; // Special case: uses different benchmark interface

    static KernelConstraints standard() { return {false, 0, false}; }

    static KernelConstraints tensorCore() { return {true, 16, false}; }

    static KernelConstraints tensorCoreComputeOnly() { return {true, 16, true}; }

    bool isSatisfied(int M, int K, int N, bool has_tensor_cores) const {
        if (M <= 0 || K <= 0 || N <= 0) {
            return false;
        }
        if (requires_tensor_cores && !has_tensor_cores) {
            return false;
        }
        if (dimension_alignment > 0) {
            if (M % dimension_alignment != 0 || K % dimension_alignment != 0 ||
                N % dimension_alignment != 0) {
                return false;
            }
        }
        return true;
    }
};

// ============================================================================
// Kernel Catalog Entry
// ============================================================================

/**
 * Complete metadata for a benchmarkable kernel.
 *
 * Each entry represents one step in the kernel optimization ladder.
 */
struct KernelCatalogEntry {
    std::string name;              // Display name for reports
    KernelType type;               // Standard or TensorCore
    KernelLauncher launcher;       // Launch adapter
    KernelConstraints constraints; // Runtime requirements

    /**
     * Get default verification tolerance for this kernel type.
     */
    VerifyTolerance defaultTolerance() const {
        return (type == KernelType::TensorCore) ? kTensorCoreVerifyTolerance
                                                : kStandardVerifyTolerance;
    }

    /**
     * Check if this kernel can run with given dimensions and hardware.
     */
    bool canRun(int M, int K, int N, bool has_tensor_cores) const {
        return constraints.isSatisfied(M, K, N, has_tensor_cores);
    }
};

// ============================================================================
// Kernel Catalog
// ============================================================================

/**
 * Returns the static catalog of all benchmarkable kernels.
 *
 * Preserves the original benchmark ordering:
 * 1. Standard FP32 kernels (Naive, Tiled, BankConflictFree, DoubleBuffer)
 * 2. Tensor Core end-to-end kernel (WMMA with FP32->FP16 conversion/fallback)
 *
 * Note: The Tensor Core compute-only benchmark is NOT included because it
 * has a different interface (requires cublas handle). It is handled as a
 * special case via getTensorCoreComputeOnlyEntry().
 */
inline const std::vector<KernelCatalogEntry> &getKernelCatalog() {
    static const std::vector<KernelCatalogEntry> catalog = {
        // Standard FP32 kernels - no constraints
        {"Naive", KernelType::Standard,
         [](const float *A, const float *B, float *C, int M, int K, int N) {
             launch_naive_sgemm<32>(A, B, C, M, K, N);
         },
         KernelConstraints::standard()},
        {"Tiled (32x32)", KernelType::Standard,
         [](const float *A, const float *B, float *C, int M, int K, int N) {
             launch_tiled_sgemm<32>(A, B, C, M, K, N);
         },
         KernelConstraints::standard()},
        {"Bank Conflict Free", KernelType::Standard,
         [](const float *A, const float *B, float *C, int M, int K, int N) {
             launch_bank_conflict_free_sgemm<32>(A, B, C, M, K, N);
         },
         KernelConstraints::standard()},
        {"Double Buffer", KernelType::Standard,
         [](const float *A, const float *B, float *C, int M, int K, int N) {
             launch_double_buffer_sgemm<32>(A, B, C, M, K, N);
         },
         KernelConstraints::standard()},
        // Tensor Core end-to-end kernel - requires sm_70+ and 16-aligned dimensions
        {"Tensor Core (WMMA end-to-end)", KernelType::TensorCore,
         [](const float *A, const float *B, float *C, int M, int K, int N) {
             launch_tensor_core_sgemm_with_fallback(A, B, C, M, K, N, defaultTensorCoreFallback());
         },
         KernelConstraints::tensorCore()}};

    return catalog;
}

/**
 * Returns the Tensor Core compute-only entry.
 *
 * This is a special entry that uses a different benchmark interface
 * (requires cublas handle for reference computation).
 */
inline KernelCatalogEntry getTensorCoreComputeOnlyEntry() {
    return {"Tensor Core (WMMA compute-only)", KernelType::TensorCore,
            nullptr, // Launcher is not used for compute-only; handled specially
            KernelConstraints::tensorCoreComputeOnly()};
}

// ============================================================================
// Catalog Query Utilities
// ============================================================================

/**
 * Count kernels by type in the catalog.
 */
inline int countKernelsByType(KernelType type) {
    int count = 0;
    for (const auto &entry : getKernelCatalog()) {
        if (entry.type == type) {
            count++;
        }
    }
    return count;
}

/**
 * Get list of kernel names for a given type.
 */
inline std::vector<std::string> getKernelNames(KernelType type) {
    std::vector<std::string> names;
    for (const auto &entry : getKernelCatalog()) {
        if (entry.type == type) {
            names.push_back(entry.name);
        }
    }
    return names;
}

/**
 * Check if any Tensor Core kernel can run with given dimensions.
 */
inline bool canRunTensorCoreKernels(int M, int K, int N, bool has_tensor_cores) {
    for (const auto &entry : getKernelCatalog()) {
        if (entry.type == KernelType::TensorCore) {
            if (entry.canRun(M, K, N, has_tensor_cores)) {
                return true;
            }
        }
    }
    return false;
}
