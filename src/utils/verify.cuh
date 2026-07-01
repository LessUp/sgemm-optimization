#pragma once

#include "cuda_utils.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

// ============================================================================
// Verification Module
//
// Provides unified verification strategy for SGEMM results:
// - Tolerance configuration (FP32 standard, Tensor Core relaxed)
// - Matrix comparison with numpy-style allclose semantics
// - cuBLAS reference computation adapter
//
// Design:
// - Single source of truth for tolerance policies
// - Separates reference computation from comparison logic
// - Supports both device and host pointer comparisons
// ============================================================================

// ============================================================================
// Verification Result Structure
// ============================================================================

/**
 * Result of matrix verification
 *
 * Captures all metrics needed to assess correctness and diagnose issues.
 */
struct VerifyResult {
    bool passed;
    float max_abs_error;
    float max_rel_error;
    int error_count;
    size_t total_elements;

    void print(const char *kernel_name = "Kernel") const {
        printf("  %s Verification: %s\n", kernel_name, passed ? "PASSED" : "FAILED");
        printf("    Max Absolute Error: %.6e\n", max_abs_error);
        printf("    Max Relative Error: %.6e\n", max_rel_error);
        if (!passed) {
            printf("    Error Count: %d / %zu (%.2f%%)\n", error_count, total_elements,
                   100.0f * error_count / total_elements);
        }
    }

    /**
     * Legacy compatibility: returns true if verification failed.
     * Used by older code that checks shouldFlagAsIncorrect().
     */
    bool shouldFlagAsIncorrect() const { return !passed; }
};

// ============================================================================
// Tolerance Configuration
// ============================================================================

/**
 * Tolerance specification for matrix comparison.
 *
 * Uses numpy-style allclose semantics:
 * |test - ref| <= atol + rtol * |ref|
 */
struct VerifyTolerance {
    float rtol; // Relative tolerance
    float atol; // Absolute tolerance
};

// Standard verification tolerance for FP32 kernels
inline constexpr VerifyTolerance kStandardVerifyTolerance{1e-3f, 1e-4f};

// Tensor Core verification tolerance (FP16 intermediate precision)
inline constexpr VerifyTolerance kTensorCoreVerifyTolerance{5e-2f, 1e-2f};

/**
 * Compute tolerance threshold for a specific reference value.
 */
inline float toleranceForValue(float ref_val, VerifyTolerance tolerance) {
    return tolerance.atol + tolerance.rtol * std::fabs(ref_val);
}

/**
 * Check if a test value is within tolerance of a reference value.
 */
inline bool isWithinTolerance(float test_val, float ref_val, VerifyTolerance tolerance) {
    float abs_error = std::fabs(test_val - ref_val);
    return abs_error <= toleranceForValue(ref_val, tolerance);
}

// ============================================================================
// Matrix Comparison (Internal Implementation)
// ============================================================================

namespace detail {

/**
 * Internal: Compare two host matrices and return verification result.
 *
 * This is the core comparison logic used by all verification functions.
 */
inline VerifyResult compareMatricesImpl(const float *h_test, const float *h_ref,
                                        size_t num_elements, VerifyTolerance tolerance) {
    VerifyResult result;
    result.max_abs_error = 0.0f;
    result.max_rel_error = 0.0f;
    result.error_count = 0;
    result.total_elements = num_elements;

    for (size_t i = 0; i < num_elements; ++i) {
        float ref_val = h_ref[i];
        float test_val = h_test[i];

        // Check for NaN or Inf in test output
        if (std::isnan(test_val) || std::isinf(test_val)) {
            // If reference is also the same NaN/Inf, consider it a match
            if (std::isnan(ref_val) && std::isnan(test_val)) {
                continue;
            }
            if (std::isinf(ref_val) && std::isinf(test_val) &&
                std::signbit(ref_val) == std::signbit(test_val)) {
                continue;
            }
            result.error_count++;
            result.max_abs_error = std::numeric_limits<float>::infinity();
            result.max_rel_error = std::numeric_limits<float>::infinity();
            continue;
        }

        // Skip comparison when reference is NaN/Inf (can't meaningfully compare)
        if (std::isnan(ref_val) || std::isinf(ref_val)) {
            result.error_count++;
            result.max_abs_error = std::numeric_limits<float>::infinity();
            result.max_rel_error = std::numeric_limits<float>::infinity();
            continue;
        }

        float abs_error = std::fabs(test_val - ref_val);
        float rel_error = abs_error / (std::fabs(ref_val) + 1e-8f);

        result.max_abs_error = std::max(result.max_abs_error, abs_error);
        result.max_rel_error = std::max(result.max_rel_error, rel_error);

        if (!isWithinTolerance(test_val, ref_val, tolerance)) {
            result.error_count++;
        }
    }

    result.passed = (result.error_count == 0);
    return result;
}

} // namespace detail

// ============================================================================
// Standalone Verification Functions
// ============================================================================

/**
 * Compare two host matrices and return verification result.
 *
 * @param h_test Test matrix (host pointer)
 * @param h_ref Reference matrix (host pointer)
 * @param M, N Matrix dimensions
 * @param tolerance Verification tolerance
 */
inline VerifyResult compareMatrices(const float *h_test, const float *h_ref, int M, int N,
                                    VerifyTolerance tolerance = kStandardVerifyTolerance) {
    size_t num_elements = static_cast<size_t>(M) * N;
    return detail::compareMatricesImpl(h_test, h_ref, num_elements, tolerance);
}

/**
 * Compare two device matrices and return verification result.
 *
 * Copies both matrices to host before comparison.
 *
 * @param d_test Test matrix (device pointer)
 * @param d_ref Reference matrix (device pointer)
 * @param M, N Matrix dimensions
 * @param tolerance Verification tolerance
 */
inline VerifyResult compareDeviceMatrices(const float *d_test, const float *d_ref, int M, int N,
                                          VerifyTolerance tolerance = kStandardVerifyTolerance) {
    size_t num_elements = static_cast<size_t>(M) * N;
    std::vector<float> h_test(num_elements);
    std::vector<float> h_ref(num_elements);

    CUDA_CHECK(
        cudaMemcpy(h_test.data(), d_test, num_elements * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(h_ref.data(), d_ref, num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    return detail::compareMatricesImpl(h_test.data(), h_ref.data(), num_elements, tolerance);
}

// ============================================================================
// cuBLAS Reference Provider
// ============================================================================

/**
 * RAII wrapper for cuBLAS-based reference computation.
 *
 * Provides:
 * - cuBLAS handle management
 * - Reference SGEMM computation
 * - Verification against kernel output
 *
 * This is the primary adapter for producing reference results.
 */
class SGEMMVerifier {
  public:
    SGEMMVerifier() { CUBLAS_CHECK(cublasCreate(&handle_)); }

    ~SGEMMVerifier() {
        if (handle_) {
            cublasDestroy(handle_);
        }
    }

    // Non-copyable, non-movable
    SGEMMVerifier(const SGEMMVerifier &) = delete;
    SGEMMVerifier &operator=(const SGEMMVerifier &) = delete;

    /**
     * Compute reference result using cuBLAS.
     *
     * C = alpha * A * B + beta * C
     * A: M x K, B: K x N, C: M x N (row-major)
     *
     * @param d_A, d_B Input matrices (device pointers)
     * @param d_C Output matrix (device pointer)
     * @param M, K, N Matrix dimensions
     * @param alpha, beta Scaling factors
     */
    void computeReference(const float *d_A, const float *d_B, float *d_C, int M, int K, int N,
                          float alpha = 1.0f, float beta = 0.0f) {
        // cuBLAS uses column-major, so we compute C^T = B^T * A^T
        // which gives us C in row-major format
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K,
                                 &beta, d_C, N));
    }

    /**
     * Verify kernel output against reference (host pointers).
     */
    VerifyResult verify(const float *h_test, const float *h_ref, int M, int N,
                        VerifyTolerance tolerance = kStandardVerifyTolerance) {
        return compareMatrices(h_test, h_ref, M, N, tolerance);
    }

    /**
     * Verify kernel output against reference (device pointers).
     */
    VerifyResult verifyDevice(const float *d_test, const float *d_ref, int M, int N,
                              VerifyTolerance tolerance = kStandardVerifyTolerance) {
        return compareDeviceMatrices(d_test, d_ref, M, N, tolerance);
    }

    /**
     * Access the underlying cuBLAS handle.
     *
     * Note: This is provided for compatibility with existing code that needs
     * direct cuBLAS access (e.g., Tensor Core compute-only benchmark).
     * Prefer using computeReference() for standard verification flows.
     */
    cublasHandle_t getHandle() { return handle_; }

  private:
    cublasHandle_t handle_;
};

// ============================================================================
// Legacy Compatibility
// ============================================================================

/**
 * Legacy function: Check if verification result indicates failure.
 *
 * @deprecated Use VerifyResult::shouldFlagAsIncorrect() or check result.passed directly.
 */
[[deprecated("Use VerifyResult::shouldFlagAsIncorrect() or check result.passed directly")]]
inline bool shouldFlagAsIncorrect(const VerifyResult &result) {
    return !result.passed;
}
