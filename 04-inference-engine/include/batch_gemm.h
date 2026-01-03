#pragma once

#include "common.h"
#include "kernels.cuh"
#include "stream_manager.h"
#include <vector>

namespace mini_inference {

// ============================================================================
// Batched GEMM Operations
// Supports multiple matrix multiplications in parallel
// ============================================================================

struct BatchGemmDesc {
    std::vector<const float*> A_ptrs;
    std::vector<const float*> B_ptrs;
    std::vector<float*> C_ptrs;
    int M, N, K;
    int batch_size;
    
    BatchGemmDesc(int m, int n, int k, int batch)
        : M(m), N(n), K(k), batch_size(batch) {
        A_ptrs.reserve(batch);
        B_ptrs.reserve(batch);
        C_ptrs.reserve(batch);
    }
    
    void add_matrices(const float* A, const float* B, float* C) {
        A_ptrs.push_back(A);
        B_ptrs.push_back(B);
        C_ptrs.push_back(C);
    }
};

// Batched GEMM using multiple streams
inline void launch_batched_gemm_streams(
    const BatchGemmDesc& desc,
    GemmKernelType kernel_type = GemmKernelType::REGISTER_BLOCKED,
    cublasHandle_t cublas_handle = nullptr)
{
    StreamManager& sm = StreamManager::instance();
    
    for (int i = 0; i < desc.batch_size; i++) {
        cudaStream_t stream = sm.get_stream();
        
        switch (kernel_type) {
            case GemmKernelType::NAIVE:
                launch_naive_matmul(desc.A_ptrs[i], desc.B_ptrs[i], desc.C_ptrs[i],
                                   desc.M, desc.N, desc.K, stream);
                break;
            case GemmKernelType::TILED:
                launch_tiled_gemm(desc.A_ptrs[i], desc.B_ptrs[i], desc.C_ptrs[i],
                                 desc.M, desc.N, desc.K, stream);
                break;
            case GemmKernelType::COALESCED:
                launch_coalesced_gemm(desc.A_ptrs[i], desc.B_ptrs[i], desc.C_ptrs[i],
                                     desc.M, desc.N, desc.K, stream);
                break;
            case GemmKernelType::DOUBLE_BUFFER:
                launch_double_buffer_gemm(desc.A_ptrs[i], desc.B_ptrs[i], desc.C_ptrs[i],
                                         desc.M, desc.N, desc.K, stream);
                break;
            case GemmKernelType::REGISTER_BLOCKED:
                launch_optimized_gemm(desc.A_ptrs[i], desc.B_ptrs[i], desc.C_ptrs[i],
                                     desc.M, desc.N, desc.K, stream);
                break;
            case GemmKernelType::CUBLAS:
                if (cublas_handle) {
                    cublasSetStream(cublas_handle, stream);
                    launch_cublas_gemm(cublas_handle, desc.A_ptrs[i], desc.B_ptrs[i],
                                      desc.C_ptrs[i], desc.M, desc.N, desc.K, stream);
                }
                break;
            default:
                launch_optimized_gemm(desc.A_ptrs[i], desc.B_ptrs[i], desc.C_ptrs[i],
                                     desc.M, desc.N, desc.K, stream);
        }
    }
    
    sm.sync_all();
}

// Strided batched GEMM (matrices stored contiguously)
inline void launch_strided_batched_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K, int batch_size,
    GemmKernelType kernel_type = GemmKernelType::REGISTER_BLOCKED,
    cublasHandle_t cublas_handle = nullptr)
{
    size_t stride_A = static_cast<size_t>(M) * K;
    size_t stride_B = static_cast<size_t>(K) * N;
    size_t stride_C = static_cast<size_t>(M) * N;
    
    BatchGemmDesc desc(M, N, K, batch_size);
    for (int i = 0; i < batch_size; i++) {
        desc.add_matrices(A + i * stride_A, B + i * stride_B, C + i * stride_C);
    }
    
    launch_batched_gemm_streams(desc, kernel_type, cublas_handle);
}

// ============================================================================
// Batched GEMM with cuBLAS (more efficient for many small matrices)
// ============================================================================

inline void launch_cublas_batched_gemm(
    cublasHandle_t handle,
    const float** A_array, const float** B_array, float** C_array,
    int M, int N, int K, int batch_size)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // cuBLAS uses column-major, so we compute C^T = B^T * A^T
    CUBLAS_CHECK(cublasSgemmBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B_array, N,
        A_array, K,
        &beta,
        C_array, N,
        batch_size
    ));
}

// ============================================================================
// Benchmark Batched GEMM
// ============================================================================

struct BatchPerfStats {
    float total_time_ms = 0.0f;
    float avg_time_per_gemm_ms = 0.0f;
    float total_gflops = 0.0f;
    int batch_size = 0;
    
    void compute(int M, int N, int K, int batch) {
        batch_size = batch;
        avg_time_per_gemm_ms = total_time_ms / batch;
        double flops_per_gemm = 2.0 * M * N * K;
        double total_flops = flops_per_gemm * batch;
        total_gflops = static_cast<float>(total_flops / (total_time_ms * 1e6));
    }
};

inline BatchPerfStats benchmark_batched_gemm(
    const BatchGemmDesc& desc,
    GemmKernelType kernel_type,
    int warmup_iters = 3,
    int bench_iters = 10,
    cublasHandle_t cublas_handle = nullptr)
{
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        launch_batched_gemm_streams(desc, kernel_type, cublas_handle);
    }
    
    // Benchmark
    GpuTimer timer;
    timer.start();
    
    for (int i = 0; i < bench_iters; i++) {
        launch_batched_gemm_streams(desc, kernel_type, cublas_handle);
    }
    
    timer.stop();
    
    BatchPerfStats stats;
    stats.total_time_ms = timer.elapsed_ms() / bench_iters;
    stats.compute(desc.M, desc.N, desc.K, desc.batch_size);
    
    return stats;
}

} // namespace mini_inference
