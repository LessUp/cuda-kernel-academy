#pragma once

#include "common.h"

namespace mini_inference {

// ============================================================================
// Kernel Declarations
// ============================================================================

// Level 1: Naive MatMul - each thread computes one output element
__global__ void naive_matmul(const float* A, const float* B, float* C,
                              int M, int N, int K);

// Level 2: Tiled GEMM - uses shared memory
__global__ void tiled_gemm(const float* A, const float* B, float* C,
                            int M, int N, int K);

// Level 3: Coalesced memory access GEMM
__global__ void coalesced_gemm(const float* A, const float* B, float* C,
                                int M, int N, int K);

// Level 4: Double buffered GEMM
__global__ void double_buffer_gemm(const float* A, const float* B, float* C,
                                    int M, int N, int K);

// Level 5: Register blocked optimized GEMM
template<int BM, int BN, int BK, int TM, int TN>
__global__ void optimized_gemm(const float* A, const float* B, float* C,
                                int M, int N, int K);

// Level 6: Fused GEMM + Bias + ReLU
template<int BM, int BN, int BK, int TM, int TN, bool ADD_BIAS, bool APPLY_RELU>
__global__ void fused_gemm_bias_relu(const float* A, const float* B, float* C,
                                      const float* bias, int M, int N, int K);

// ============================================================================
// Kernel Launch Wrappers
// ============================================================================

void launch_naive_matmul(const float* A, const float* B, float* C,
                         int M, int N, int K, cudaStream_t stream = 0);

void launch_tiled_gemm(const float* A, const float* B, float* C,
                       int M, int N, int K, cudaStream_t stream = 0);

void launch_coalesced_gemm(const float* A, const float* B, float* C,
                           int M, int N, int K, cudaStream_t stream = 0);

void launch_double_buffer_gemm(const float* A, const float* B, float* C,
                               int M, int N, int K, cudaStream_t stream = 0);

void launch_optimized_gemm(const float* A, const float* B, float* C,
                           int M, int N, int K, cudaStream_t stream = 0);

void launch_fused_gemm(const float* A, const float* B, float* C,
                       const float* bias, int M, int N, int K,
                       bool add_bias, bool apply_relu, cudaStream_t stream = 0);

// cuBLAS wrapper
void launch_cublas_gemm(cublasHandle_t handle, const float* A, const float* B, float* C,
                        int M, int N, int K, cudaStream_t stream = 0);

// ============================================================================
// Performance Measurement
// ============================================================================

class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }
    
    void stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
    }
    
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
    
private:
    cudaEvent_t start_, stop_;
};

// Benchmark a kernel with multiple iterations
PerfStats benchmark_kernel(GemmKernelType type, const float* A, const float* B, float* C,
                           int M, int N, int K, int warmup_iters = 5, int bench_iters = 20,
                           cublasHandle_t cublas_handle = nullptr, cudaStream_t stream = 0);

} // namespace mini_inference
