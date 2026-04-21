#pragma once

#include <functional>
#include <map>

#include "common.h"
#include "kernels.cuh"

namespace mini_inference {

// ============================================================================
// Auto-Tuner for GEMM Kernels
// Automatically selects the best kernel configuration for given matrix sizes
// ============================================================================

struct TuningConfig {
    int block_m;
    int block_n;
    int block_k;
    int thread_m;
    int thread_n;
    GemmKernelType kernel_type;
};

struct TuningResult {
    TuningConfig config;
    float time_ms;
    float gflops;
};

class AutoTuner {
public:
    AutoTuner() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~AutoTuner() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    // Find the best kernel for given matrix dimensions
    TuningResult tune(int M, int N, int K, cublasHandle_t cublas_handle = nullptr) {
        std::vector<TuningResult> results;

        // Allocate test matrices
        DeviceMemory d_A(M * K * sizeof(float));
        DeviceMemory d_B(K * N * sizeof(float));
        DeviceMemory d_C(M * N * sizeof(float));

        // Initialize with random data
        std::vector<float> h_A(M * K), h_B(K * N);
        random_init(h_A.data(), M * K);
        random_init(h_B.data(), K * N);
        d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
        d_B.copy_from_host(h_B.data(), K * N * sizeof(float));

        // Test each kernel type
        std::vector<GemmKernelType> kernel_types = {GemmKernelType::NAIVE,
                                                    GemmKernelType::TILED,
                                                    GemmKernelType::COALESCED,
                                                    GemmKernelType::DOUBLE_BUFFER,
                                                    GemmKernelType::REGISTER_BLOCKED,
                                                    GemmKernelType::FUSED};

        for (auto type : kernel_types) {
            TuningResult result;
            result.config.kernel_type = type;
            result.time_ms =
                benchmark_single(type, d_A.get(), d_B.get(), d_C.get(), M, N, K, cublas_handle);
            result.gflops = (2.0f * M * N * K) / (result.time_ms * 1e6f);
            results.push_back(result);
        }

        // Find best result
        auto best = std::min_element(
            results.begin(), results.end(),
            [](const TuningResult& a, const TuningResult& b) { return a.time_ms < b.time_ms; });

        // Cache the result
        cache_[make_key(M, N, K)] = *best;

        return *best;
    }

    // Get cached result or tune
    TuningResult get_best(int M, int N, int K, cublasHandle_t cublas_handle = nullptr) {
        auto key = make_key(M, N, K);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }
        return tune(M, N, K, cublas_handle);
    }

    // Execute with best kernel
    void execute_best(const float* A, const float* B, float* C, int M, int N, int K,
                      cudaStream_t stream = 0) {
        auto result = get_best(M, N, K);
        execute_kernel(result.config.kernel_type, A, B, C, M, N, K, stream);
    }

    // Print tuning results
    void print_cache() const {
        printf(
            "\n╔═══════════════════════════════════════════════════════════════════════════════╗"
            "\n");
        printf(
            "║                           Auto-Tuner Cache                                    ║\n");
        printf(
            "╠═══════════════════════════════════════════════════════════════════════════════╣\n");
        printf("║ %-20s │ %-16s │ %10s │ %10s ║\n", "Dimensions", "Best Kernel", "Time (ms)",
               "GFLOPS");
        printf(
            "╠═══════════════════════════════════════════════════════════════════════════════╣\n");

        for (const auto& [key, result] : cache_) {
            printf("║ %-20s │ %-16s │ %10.4f │ %10.2f ║\n", key.c_str(),
                   kernel_type_name(result.config.kernel_type), result.time_ms, result.gflops);
        }

        printf(
            "╚═══════════════════════════════════════════════════════════════════════════════╝\n");
    }

private:
    cudaEvent_t start_, stop_;
    std::map<std::string, TuningResult> cache_;

    std::string make_key(int M, int N, int K) const {
        return std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
    }

    float benchmark_single(GemmKernelType type, const float* A, const float* B, float* C, int M,
                           int N, int K, cublasHandle_t cublas_handle) {
        const int warmup = 3;
        const int iters = 10;

        // Warmup
        for (int i = 0; i < warmup; i++) {
            execute_kernel(type, A, B, C, M, N, K, 0, cublas_handle);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        CUDA_CHECK(cudaEventRecord(start_));
        for (int i = 0; i < iters; i++) {
            execute_kernel(type, A, B, C, M, N, K, 0, cublas_handle);
        }
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms / iters;
    }

    void execute_kernel(GemmKernelType type, const float* A, const float* B, float* C, int M, int N,
                        int K, cudaStream_t stream, cublasHandle_t cublas_handle = nullptr) {
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
};

}  // namespace mini_inference
