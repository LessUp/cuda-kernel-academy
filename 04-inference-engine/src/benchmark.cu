#include "kernels.cuh"

namespace mini_inference {

// ============================================================================
// cuBLAS Wrapper
// ============================================================================

void launch_cublas_gemm(cublasHandle_t handle, const float* A, const float* B, float* C,
                        int M, int N, int K, cudaStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major, so we compute C^T = B^T * A^T
    // which gives us C in row-major format
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             B, N,
                             A, K,
                             &beta,
                             C, N));
}

// ============================================================================
// Benchmark Function
// ============================================================================

PerfStats benchmark_kernel(GemmKernelType type, const float* A, const float* B, float* C,
                           int M, int N, int K, int warmup_iters, int bench_iters,
                           cublasHandle_t cublas_handle, cudaStream_t stream) {
    GpuTimer timer;
    PerfStats stats;

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        switch (type) {
            case GemmKernelType::NAIVE:
                launch_naive_matmul(A, B, C, M, N, K, stream);
                break;
            case GemmKernelType::TILED:
                launch_tiled_gemm(A, B, C, M, N, K, stream);
                break;
            case GemmKernelType::COALESCED:
                launch_coalesced_gemm(A, B, C, M, N, K, stream);
                break;
            case GemmKernelType::DOUBLE_BUFFER:
                launch_double_buffer_gemm(A, B, C, M, N, K, stream);
                break;
            case GemmKernelType::REGISTER_BLOCKED:
                launch_optimized_gemm(A, B, C, M, N, K, stream);
                break;
            case GemmKernelType::FUSED:
                launch_fused_gemm(A, B, C, nullptr, M, N, K, false, false, stream);
                break;
            case GemmKernelType::CUBLAS:
                if (cublas_handle) {
                    launch_cublas_gemm(cublas_handle, A, B, C, M, N, K, stream);
                }
                break;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark
    timer.start(stream);
    for (int i = 0; i < bench_iters; i++) {
        switch (type) {
            case GemmKernelType::NAIVE:
                launch_naive_matmul(A, B, C, M, N, K, stream);
                break;
            case GemmKernelType::TILED:
                launch_tiled_gemm(A, B, C, M, N, K, stream);
                break;
            case GemmKernelType::COALESCED:
                launch_coalesced_gemm(A, B, C, M, N, K, stream);
                break;
            case GemmKernelType::DOUBLE_BUFFER:
                launch_double_buffer_gemm(A, B, C, M, N, K, stream);
                break;
            case GemmKernelType::REGISTER_BLOCKED:
                launch_optimized_gemm(A, B, C, M, N, K, stream);
                break;
            case GemmKernelType::FUSED:
                launch_fused_gemm(A, B, C, nullptr, M, N, K, false, false, stream);
                break;
            case GemmKernelType::CUBLAS:
                if (cublas_handle) {
                    launch_cublas_gemm(cublas_handle, A, B, C, M, N, K, stream);
                }
                break;
        }
    }
    timer.stop(stream);

    stats.kernel_time_ms = timer.elapsed_ms() / bench_iters;
    stats.compute_gflops(M, N, K);

    // Memory bandwidth: read A (M*K) + B (K*N), write C (M*N)
    double bytes = (static_cast<double>(M) * K + static_cast<double>(K) * N +
                   static_cast<double>(M) * N) * sizeof(float);
    stats.memory_bandwidth_gb = static_cast<float>(bytes / (stats.kernel_time_ms * 1e6));

    return stats;
}

} // namespace mini_inference
