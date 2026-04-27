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
// Verification Result Structure
// ============================================================================

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
};

struct VerifyTolerance {
    float rtol;
    float atol;
};

inline constexpr VerifyTolerance kStandardVerifyTolerance{1e-3f, 1e-4f};
inline constexpr VerifyTolerance kTensorCoreVerifyTolerance{5e-2f, 1e-2f};

inline float toleranceForValue(float ref_val, VerifyTolerance tolerance) {
    return tolerance.atol + tolerance.rtol * std::fabs(ref_val);
}

inline bool isWithinTolerance(float test_val, float ref_val, VerifyTolerance tolerance) {
    float abs_error = std::fabs(test_val - ref_val);
    return abs_error <= toleranceForValue(ref_val, tolerance);
}

// ============================================================================
// cuBLAS Reference SGEMM
// ============================================================================

class SGEMMVerifier {
  public:
    SGEMMVerifier() { CUBLAS_CHECK(cublasCreate(&handle_)); }

    ~SGEMMVerifier() {
        if (handle_) {
            cublasDestroy(handle_);
        }
    }

    // Compute reference result using cuBLAS
    // C = alpha * A * B + beta * C
    // A: M x K, B: K x N, C: M x N (row-major)
    void computeReference(const float *d_A, const float *d_B, float *d_C, int M, int K, int N,
                          float alpha = 1.0f, float beta = 0.0f) {
        // cuBLAS uses column-major, so we compute C^T = B^T * A^T
        // which gives us C in row-major format
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K,
                                 &beta, d_C, N));
    }

    // Verify kernel output against reference
    // Uses numpy-style allclose: |test - ref| <= atol + rtol * |ref|
    VerifyResult verify(const float *h_test, const float *h_ref, int M, int N,
                        VerifyTolerance tolerance = kStandardVerifyTolerance) {
        VerifyResult result;
        result.max_abs_error = 0.0f;
        result.max_rel_error = 0.0f;
        result.error_count = 0;
        result.total_elements = M * N;

        for (int i = 0; i < M * N; ++i) {
            float ref_val = h_ref[i];
            float test_val = h_test[i];

            if (std::isnan(test_val) || std::isinf(test_val)) {
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

    // Verify with device pointers (copies to host internally)
    VerifyResult verifyDevice(const float *d_test, const float *d_ref, int M, int N,
                              VerifyTolerance tolerance = kStandardVerifyTolerance) {
        size_t num_elements = static_cast<size_t>(M) * N;
        std::vector<float> h_test(num_elements);
        std::vector<float> h_ref(num_elements);

        CUDA_CHECK(cudaMemcpy(h_test.data(), d_test, num_elements * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(
            cudaMemcpy(h_ref.data(), d_ref, num_elements * sizeof(float), cudaMemcpyDeviceToHost));

        return verify(h_test.data(), h_ref.data(), M, N, tolerance);
    }

    // Keep error flagging semantics aligned with compareMatrices/verify.
    static bool shouldFlagAsIncorrect(const VerifyResult &result) { return !result.passed; }

    cublasHandle_t getHandle() { return handle_; }

  private:
    cublasHandle_t handle_;
};

// ============================================================================
// Standalone Verification Functions
// ============================================================================

// Compare two matrices and return verification result
// Uses numpy-style allclose: |test - ref| <= atol + rtol * |ref|
inline VerifyResult compareMatrices(const float *h_test, const float *h_ref, int M, int N,
                                    VerifyTolerance tolerance = kStandardVerifyTolerance) {
    VerifyResult result;
    result.max_abs_error = 0.0f;
    result.max_rel_error = 0.0f;
    result.error_count = 0;
    result.total_elements = static_cast<size_t>(M) * N;

    for (size_t i = 0; i < static_cast<size_t>(M) * N; ++i) {
        float ref_val = h_ref[i];
        float test_val = h_test[i];

        // Check for NaN or Inf
        if (std::isnan(test_val) || std::isinf(test_val)) {
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
