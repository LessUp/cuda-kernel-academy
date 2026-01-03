#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include "cuda_utils.cuh"

// ============================================================================
// Verification Result Structure
// ============================================================================

struct VerifyResult {
    bool passed;
    float max_abs_error;
    float max_rel_error;
    int error_count;
    int total_elements;
    
    void print(const char* kernel_name = "Kernel") const {
        printf("  %s Verification: %s\n", kernel_name, passed ? "PASSED" : "FAILED");
        printf("    Max Absolute Error: %.6e\n", max_abs_error);
        printf("    Max Relative Error: %.6e\n", max_rel_error);
        if (!passed) {
            printf("    Error Count: %d / %d (%.2f%%)\n", 
                   error_count, total_elements, 
                   100.0f * error_count / total_elements);
        }
    }
};

// ============================================================================
// cuBLAS Reference SGEMM
// ============================================================================

class SGEMMVerifier {
public:
    SGEMMVerifier() {
        CUBLAS_CHECK(cublasCreate(&handle_));
    }
    
    ~SGEMMVerifier() {
        if (handle_) {
            cublasDestroy(handle_);
        }
    }
    
    // Compute reference result using cuBLAS
    // C = alpha * A * B + beta * C
    // A: M x K, B: K x N, C: M x N (row-major)
    void computeReference(const float* d_A, const float* d_B, float* d_C,
                          int M, int K, int N,
                          float alpha = 1.0f, float beta = 0.0f) {
        // cuBLAS uses column-major, so we compute C^T = B^T * A^T
        // which gives us C in row-major format
        CUBLAS_CHECK(cublasSgemm(handle_,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 d_B, N,
                                 d_A, K,
                                 &beta,
                                 d_C, N));
    }
    
    // Verify kernel output against reference
    // Uses numpy-style allclose: |test - ref| <= atol + rtol * |ref|
    VerifyResult verify(const float* h_test, const float* h_ref,
                        int M, int N,
                        float rtol = 1e-4f,
                        float atol = 1e-5f) {
        VerifyResult result;
        result.max_abs_error = 0.0f;
        result.max_rel_error = 0.0f;
        result.error_count = 0;
        result.total_elements = M * N;
        
        for (int i = 0; i < M * N; ++i) {
            float ref_val = h_ref[i];
            float test_val = h_test[i];
            
            float abs_error = std::fabs(test_val - ref_val);
            float rel_error = abs_error / (std::fabs(ref_val) + 1e-8f);
            
            result.max_abs_error = std::max(result.max_abs_error, abs_error);
            result.max_rel_error = std::max(result.max_rel_error, rel_error);
            
            // numpy-style allclose: |test - ref| <= atol + rtol * |ref|
            float tolerance = atol + rtol * std::fabs(ref_val);
            if (abs_error > tolerance) {
                result.error_count++;
            }
        }
        
        result.passed = (result.error_count == 0);
        
        return result;
    }
    
    // Verify with device pointers (copies to host internally)
    VerifyResult verifyDevice(const float* d_test, const float* d_ref,
                              int M, int N,
                              float rtol = 1e-4f,
                              float atol = 1e-5f) {
        std::vector<float> h_test(M * N);
        std::vector<float> h_ref(M * N);
        
        CUDA_CHECK(cudaMemcpy(h_test.data(), d_test, M * N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, M * N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        
        return verify(h_test.data(), h_ref.data(), M, N, rtol, atol);
    }
    
    // Check if result should be flagged as incorrect based on threshold
    static bool shouldFlagAsIncorrect(float max_rel_error, bool is_tensor_core) {
        float threshold = is_tensor_core ? 1e-3f : 1e-4f;
        return max_rel_error > threshold;
    }
    
    cublasHandle_t getHandle() { return handle_; }
    
private:
    cublasHandle_t handle_;
};

// ============================================================================
// Standalone Verification Functions
// ============================================================================

// Compare two matrices and return verification result
// Uses numpy-style allclose: |test - ref| <= atol + rtol * |ref|
inline VerifyResult compareMatrices(const float* h_test, const float* h_ref,
                                    int M, int N,
                                    float rtol = 1e-4f,
                                    float atol = 1e-5f) {
    VerifyResult result;
    result.max_abs_error = 0.0f;
    result.max_rel_error = 0.0f;
    result.error_count = 0;
    result.total_elements = M * N;
    
    for (int i = 0; i < M * N; ++i) {
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
        
        // numpy-style allclose: |test - ref| <= atol + rtol * |ref|
        float tolerance = atol + rtol * std::fabs(ref_val);
        if (abs_error > tolerance) {
            result.error_count++;
        }
    }
    
    // Pass if no elements exceed tolerance
    result.passed = (result.error_count == 0);
    
    return result;
}

// Quick check if matrices are approximately equal
inline bool matricesApproxEqual(const float* h_test, const float* h_ref,
                                int M, int N,
                                float rtol = 1e-4f,
                                float atol = 1e-5f) {
    for (int i = 0; i < M * N; ++i) {
        float abs_error = std::fabs(h_test[i] - h_ref[i]);
        float rel_error = abs_error / (std::fabs(h_ref[i]) + 1e-8f);
        
        if (abs_error > atol && rel_error > rtol) {
            return false;
        }
    }
    return true;
}

// Find first mismatch location (for debugging)
inline int findFirstMismatch(const float* h_test, const float* h_ref,
                             int M, int N,
                             float rtol = 1e-4f,
                             float atol = 1e-5f) {
    for (int i = 0; i < M * N; ++i) {
        float abs_error = std::fabs(h_test[i] - h_ref[i]);
        float rel_error = abs_error / (std::fabs(h_ref[i]) + 1e-8f);
        
        if (abs_error > atol && rel_error > rtol) {
            return i;
        }
    }
    return -1;  // No mismatch found
}
