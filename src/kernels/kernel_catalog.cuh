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
// Centralizes kernel metadata, launch adapters, and kernel type classification.
// Eliminates repeated lambdas and inline dispatch logic in BenchmarkRunner.
//
// Design:
// - Static catalog of standard kernels (no global mutable state)
// - Each entry contains: name, type, and launch adapter
// - BenchmarkRunner iterates over catalog entries instead of hardcoding dispatch
//
// Note: Tensor Core compute-only benchmark remains a special case due to
// its different interface (requires cublas handle), but the end-to-end
// tensor core kernel is included in the catalog.
// ============================================================================

// ============================================================================
// Kernel Launch Adapter Type
// ============================================================================

using KernelLauncher = std::function<void(const float*, const float*, float*, int, int, int)>;

// ============================================================================
// Kernel Catalog Entry
// ============================================================================

struct KernelCatalogEntry {
    std::string name;
    KernelType type;
    KernelLauncher launcher;
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
 * has a different interface (requires cublas handle) and is handled separately
 * in BenchmarkRunner.
 */
inline const std::vector<KernelCatalogEntry>& getKernelCatalog() {
    static const std::vector<KernelCatalogEntry> catalog = {
        // Standard FP32 kernels
        {
            "Naive SGEMM",
            KernelType::Standard,
            [](const float *A, const float *B, float *C, int M, int K, int N) {
                launch_naive_sgemm<32>(A, B, C, M, K, N);
            }
        },
        {
            "Tiled SGEMM",
            KernelType::Standard,
            [](const float *A, const float *B, float *C, int M, int K, int N) {
                launch_tiled_sgemm<32>(A, B, C, M, K, N);
            }
        },
        {
            "Bank Conflict Free SGEMM",
            KernelType::Standard,
            [](const float *A, const float *B, float *C, int M, int K, int N) {
                launch_bank_conflict_free_sgemm<32>(A, B, C, M, K, N);
            }
        },
        {
            "Double Buffer SGEMM",
            KernelType::Standard,
            [](const float *A, const float *B, float *C, int M, int K, int N) {
                launch_double_buffer_sgemm<32>(A, B, C, M, K, N);
            }
        },
        // Tensor Core end-to-end kernel
        {
            "Tensor Core SGEMM (end-to-end",
            KernelType::TensorCore,
            [](const float *A, const float *B, float *C, int M, int K, int N) {
                launch_tensor_core_sgemm_with_fallback(A, B, C, M, K, N,
                                                       defaultTensorCoreFallback());
            }
        }
    };
    
    return catalog;
}
