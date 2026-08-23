#pragma once

/**
 * @file common.h
 * @brief Common utilities for mini_inference module
 *
 * This file provides common utilities and backward-compatible aliases
 * for the mini_inference module. Core CUDA utilities are now provided
 * by cuda_foundations in common/.
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Include common cuda_foundations utilities
#include <cuda_foundations/cuda_foundations.hpp>

namespace mini_inference {

// ============================================================================
// CUDA Error Handling (backward compatible)
// ============================================================================

// Use cuda_foundations's CudaError as CudaException for compatibility
using CudaException = cuda_foundations::core::CudaError;

// Backward-compatible macro
#define CUDA_CHECK(call) CA_CUDA_CHECK(call)

// cuBLAS check (kept here as it's not in cuda_foundations)
#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        cublasStatus_t status = call;                                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            throw std::runtime_error("cuBLAS error: " + std::to_string(status)); \
        }                                                                        \
    } while (0)

// ============================================================================
// Data Structures
// ============================================================================

struct MatrixDesc {
    float* data = nullptr;
    int rows = 0;
    int cols = 0;
    int ld = 0;  // leading dimension
    bool is_transposed = false;

    MatrixDesc() = default;
    MatrixDesc(float* d, int r, int c, int l = 0, bool t = false)
        : data(d), rows(r), cols(c), ld(l ? l : c), is_transposed(t) {}

    size_t size_bytes() const { return static_cast<size_t>(rows) * cols * sizeof(float); }
    size_t num_elements() const { return static_cast<size_t>(rows) * cols; }
};

struct GemmConfig {
    int BLOCK_M = 128;
    int BLOCK_N = 128;
    int BLOCK_K = 8;
    int WARP_M = 32;
    int WARP_N = 64;
    bool use_double_buffer = true;
    bool use_vectorized_load = true;
};

struct FusionConfig {
    bool add_bias = false;
    bool apply_relu = false;
    float* bias = nullptr;
};

struct PerfStats {
    float kernel_time_ms = 0.0f;
    float gflops = 0.0f;
    float memory_bandwidth_gb = 0.0f;
    float cublas_ratio = 0.0f;

    void compute_gflops(int M, int N, int K) {
        // GEMM: 2*M*N*K FLOPs
        double flops = 2.0 * M * N * K;
        gflops = static_cast<float>(flops / (kernel_time_ms * 1e6));
    }
};

enum class GemmKernelType {
    NAIVE,
    TILED,
    COALESCED,
    DOUBLE_BUFFER,
    REGISTER_BLOCKED,
    FUSED,
    CUBLAS
};

inline const char* kernel_type_name(GemmKernelType type) {
    switch (type) {
        case GemmKernelType::NAIVE:
            return "Naive";
        case GemmKernelType::TILED:
            return "Tiled";
        case GemmKernelType::COALESCED:
            return "Coalesced";
        case GemmKernelType::DOUBLE_BUFFER:
            return "DoubleBuffer";
        case GemmKernelType::REGISTER_BLOCKED:
            return "RegisterBlocked";
        case GemmKernelType::FUSED:
            return "Fused";
        case GemmKernelType::CUBLAS:
            return "cuBLAS";
        default:
            return "Unknown";
    }
}

// ============================================================================
// Device Memory RAII Wrapper
// ============================================================================

class DeviceMemory {
public:
    DeviceMemory() = default;

    explicit DeviceMemory(size_t bytes) { allocate(bytes); }

    ~DeviceMemory() { free(); }

    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    DeviceMemory(DeviceMemory&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            free();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void allocate(size_t bytes) {
        if (size_ == bytes && ptr_ != nullptr) {
            return;
        }

        free();
        if (bytes == 0) {
            return;
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), bytes));
        size_ = bytes;
    }

    void free() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    float* get() noexcept { return ptr_; }
    const float* get() const noexcept { return ptr_; }
    size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return ptr_ == nullptr; }

    void copy_from_host(const void* host_data, size_t bytes) {
        CUDA_CHECK(cudaMemcpy(ptr_, host_data, bytes, cudaMemcpyHostToDevice));
    }

    void copy_to_host(void* host_data, size_t bytes) const {
        CUDA_CHECK(cudaMemcpy(host_data, ptr_, bytes, cudaMemcpyDeviceToHost));
    }

    void zero() {
        if (ptr_ != nullptr && size_ > 0) {
            CUDA_CHECK(cudaMemset(ptr_, 0, size_));
        }
    }

private:
    float* ptr_ = nullptr;
    size_t size_ = 0;
};

// ============================================================================
// Utility Functions
// ============================================================================

inline void validate_gemm_inputs(const MatrixDesc& A, const MatrixDesc& B, const MatrixDesc& C) {
    // Dimension match check
    if (A.cols != B.rows) {
        std::ostringstream oss;
        oss << "Matrix dimension mismatch: A(" << A.rows << "," << A.cols << ") × B(" << B.rows
            << "," << B.cols << ") - "
            << "A.cols(" << A.cols << ") != B.rows(" << B.rows << ")";
        throw std::invalid_argument(oss.str());
    }

    // Output dimension check
    if (C.rows != A.rows || C.cols != B.cols) {
        std::ostringstream oss;
        oss << "Output matrix dimension mismatch: expected (" << A.rows << "," << B.cols
            << "), got (" << C.rows << "," << C.cols << ")";
        throw std::invalid_argument(oss.str());
    }

    // Null pointer check
    if (!A.data || !B.data || !C.data) {
        throw std::invalid_argument("Null pointer in matrix data");
    }
}

inline void validate_gemm_dimensions(int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) {
        std::ostringstream oss;
        oss << "Invalid matrix dimensions: M=" << M << ", N=" << N << ", K=" << K;
        throw std::invalid_argument(oss.str());
    }
}

// CPU reference implementation for validation
inline void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// CPU reference with bias and ReLU
inline void cpu_matmul_bias_relu(const float* A, const float* B, float* C, const float* bias, int M,
                                 int N, int K, bool add_bias, bool apply_relu) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            if (add_bias && bias) {
                sum += bias[j];
            }
            if (apply_relu) {
                sum = std::max(0.0f, sum);
            }
            C[i * N + j] = sum;
        }
    }
}

// Compare matrices and return max absolute error
inline float compare_matrices(const float* A, const float* B, size_t n) {
    float max_error = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float error = std::abs(A[i] - B[i]);
        max_error = std::max(max_error, error);
    }
    return max_error;
}

// Random matrix initialization
inline void random_init(float* data, size_t n, float min_val = -1.0f, float max_val = 1.0f) {
    // Function-local static generator with a fixed seed for reproducibility;
    // the mutex makes concurrent calls from multiple threads safe.
    static std::mt19937 gen(42);
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (size_t i = 0; i < n; i++) {
        data[i] = dist(gen);
    }
}

// Zero initialization
inline void zero_init(float* data, size_t n) {
    std::fill(data, data + n, 0.0f);
}

}  // namespace mini_inference
