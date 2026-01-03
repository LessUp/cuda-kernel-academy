#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <string>
#include <functional>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include "cuda_utils.cuh"
#include "verify.cuh"

// ============================================================================
// Benchmark Result Structure
// ============================================================================

struct BenchmarkResult {
    std::string kernel_name;
    int M, K, N;
    float time_ms;
    float gflops;
    float bandwidth_gb_s;
    bool correct;
    float max_error;
    float efficiency;  // Percentage of theoretical peak
    
    void print() const {
        printf("  %-25s | %4d x %4d x %4d | %8.3f ms | %8.2f GFLOPS | %s | err: %.2e\n",
               kernel_name.c_str(), M, K, N, time_ms, gflops,
               correct ? "PASS" : "FAIL", max_error);
    }
};

// ============================================================================
// SGEMM Benchmark Class
// ============================================================================

class SGEMMBenchmark {
public:
    SGEMMBenchmark() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    }
    
    ~SGEMMBenchmark() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
        cublasDestroy(cublas_handle_);
    }
    
    /**
     * Run benchmark for a single kernel
     * 
     * @param name Kernel name for reporting
     * @param kernel_func Function that launches the kernel
     * @param M, K, N Matrix dimensions
     * @param warmup_runs Number of warm-up iterations
     * @param benchmark_runs Number of timed iterations
     * @param rtol Relative tolerance for correctness check
     * @param atol Absolute tolerance for correctness check
     * @return BenchmarkResult with timing and correctness info
     */
    template<typename KernelFunc>
    BenchmarkResult run(
        const std::string& name,
        KernelFunc kernel_func,
        int M, int K, int N,
        int warmup_runs = 5,
        int benchmark_runs = 20,
        float rtol = 1e-4f,
        float atol = 1e-5f
    ) {
        BenchmarkResult result;
        result.kernel_name = name;
        result.M = M;
        result.K = K;
        result.N = N;
        
        // Allocate memory
        std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_C_ref(M * N);
        float *d_A, *d_B, *d_C, *d_C_ref;
        
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_ref, M * N * sizeof(float)));
        
        // Initialize with random data
        initRandomMatrix(h_A.data(), M, K, -1.0f, 1.0f, 42);
        initRandomMatrix(h_B.data(), K, N, -1.0f, 1.0f, 123);
        
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
        
        // Compute reference using cuBLAS
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(cublas_handle_,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha, d_B, N, d_A, K,
                                 &beta, d_C_ref, N));
        
        // Warm-up runs
        for (int i = 0; i < warmup_runs; ++i) {
            CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
            kernel_func(d_A, d_B, d_C, M, K, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark runs
        CUDA_CHECK(cudaEventRecord(start_));
        for (int i = 0; i < benchmark_runs; ++i) {
            kernel_func(d_A, d_B, d_C, M, K, N);
        }
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        
        float total_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&total_time_ms, start_, stop_));
        result.time_ms = total_time_ms / benchmark_runs;
        
        // Calculate GFLOPS
        // GEMM: 2 * M * N * K FLOPs (multiply + add for each output element)
        double flops = 2.0 * M * N * K;
        result.gflops = (flops / (result.time_ms * 1e-3)) / 1e9;
        
        // Calculate memory bandwidth (approximate)
        // Read: A (M*K) + B (K*N), Write: C (M*N)
        double bytes = (M * K + K * N + M * N) * sizeof(float);
        result.bandwidth_gb_s = (bytes / (result.time_ms * 1e-3)) / 1e9;
        
        // Verify correctness
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        
        VerifyResult verify_result = compareMatrices(h_C.data(), h_C_ref.data(), M, N, rtol, atol);
        result.correct = verify_result.passed;
        result.max_error = verify_result.max_rel_error;
        
        // Cleanup
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_C_ref);
        
        results_.push_back(result);
        return result;
    }
    
    /**
     * Run cuBLAS benchmark for comparison
     */
    BenchmarkResult runCublas(int M, int K, int N, int warmup_runs = 5, int benchmark_runs = 20) {
        BenchmarkResult result;
        result.kernel_name = "cuBLAS";
        result.M = M;
        result.K = K;
        result.N = N;
        
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        
        std::vector<float> h_A(M * K), h_B(K * N);
        initRandomMatrix(h_A.data(), M, K, -1.0f, 1.0f, 42);
        initRandomMatrix(h_B.data(), K, N, -1.0f, 1.0f, 123);
        
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
        
        float alpha = 1.0f, beta = 0.0f;
        
        // Warm-up
        for (int i = 0; i < warmup_runs; ++i) {
            CUBLAS_CHECK(cublasSgemm(cublas_handle_,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K,
                                     &alpha, d_B, N, d_A, K,
                                     &beta, d_C, N));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark
        CUDA_CHECK(cudaEventRecord(start_));
        for (int i = 0; i < benchmark_runs; ++i) {
            CUBLAS_CHECK(cublasSgemm(cublas_handle_,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K,
                                     &alpha, d_B, N, d_A, K,
                                     &beta, d_C, N));
        }
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        
        float total_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&total_time_ms, start_, stop_));
        result.time_ms = total_time_ms / benchmark_runs;
        
        double flops = 2.0 * M * N * K;
        result.gflops = (flops / (result.time_ms * 1e-3)) / 1e9;
        
        double bytes = (M * K + K * N + M * N) * sizeof(float);
        result.bandwidth_gb_s = (bytes / (result.time_ms * 1e-3)) / 1e9;
        
        result.correct = true;
        result.max_error = 0.0f;
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        
        results_.push_back(result);
        return result;
    }
    
    /**
     * Print summary of all benchmark results
     */
    void printSummary() const {
        printf("\n");
        printf("================================================================================\n");
        printf("                           SGEMM Benchmark Results\n");
        printf("================================================================================\n");
        printf("  %-25s | %-17s | %10s | %14s | %4s | %s\n",
               "Kernel", "Dimensions", "Time", "Performance", "Pass", "Max Error");
        printf("--------------------------------------------------------------------------------\n");
        
        for (const auto& result : results_) {
            result.print();
        }
        
        printf("================================================================================\n");
    }
    
    /**
     * Export results for Roofline model analysis
     */
    void exportRooflineData(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            fprintf(stderr, "Failed to open file: %s\n", filename.c_str());
            return;
        }
        
        file << "kernel,M,K,N,time_ms,gflops,bandwidth_gb_s,arithmetic_intensity\n";
        
        for (const auto& result : results_) {
            // Arithmetic intensity = FLOPs / Bytes
            double flops = 2.0 * result.M * result.N * result.K;
            double bytes = (result.M * result.K + result.K * result.N + result.M * result.N) * sizeof(float);
            double ai = flops / bytes;
            
            file << result.kernel_name << ","
                 << result.M << "," << result.K << "," << result.N << ","
                 << result.time_ms << ","
                 << result.gflops << ","
                 << result.bandwidth_gb_s << ","
                 << ai << "\n";
        }
        
        file.close();
        printf("Roofline data exported to: %s\n", filename.c_str());
    }
    
    /**
     * Get all results
     */
    const std::vector<BenchmarkResult>& getResults() const {
        return results_;
    }
    
    /**
     * Clear results
     */
    void clearResults() {
        results_.clear();
    }
    
    /**
     * Get cuBLAS handle for external use
     */
    cublasHandle_t getCublasHandle() const {
        return cublas_handle_;
    }
    
private:
    cudaEvent_t start_, stop_;
    cublasHandle_t cublas_handle_;
    std::vector<BenchmarkResult> results_;
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Print performance comparison table
 */
inline void printPerformanceComparison(const std::vector<BenchmarkResult>& results,
                                       float cublas_gflops) {
    printf("\n");
    printf("Performance Comparison (vs cuBLAS):\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("  %-25s | %14s | %10s\n", "Kernel", "GFLOPS", "% of cuBLAS");
    printf("--------------------------------------------------------------------------------\n");
    
    for (const auto& result : results) {
        float percentage = (result.gflops / cublas_gflops) * 100.0f;
        printf("  %-25s | %10.2f     | %8.1f%%\n",
               result.kernel_name.c_str(), result.gflops, percentage);
    }
    printf("--------------------------------------------------------------------------------\n");
}

/**
 * Calculate theoretical peak GFLOPS for the GPU
 */
inline float getTheoreticalPeakGflops() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    // FP32 peak = SM count * cores per SM * 2 (FMA) * clock rate
    // This is approximate; actual cores per SM varies by architecture
    int coresPerSM;
    switch (prop.major) {
        case 7: coresPerSM = 64; break;   // Volta, Turing
        case 8: coresPerSM = 64; break;   // Ampere
        case 9: coresPerSM = 128; break;  // Hopper
        default: coresPerSM = 64;
    }
    
    // Use peak clock rate (in kHz) from device properties
    // Note: clockRate is deprecated, using a reasonable estimate based on architecture
    float clockGHz = 1.5f;  // Conservative estimate for modern GPUs
    float peakGflops = prop.multiProcessorCount * coresPerSM * 2 * clockGHz * 1000;
    
    return peakGflops;
}

/**
 * Calculate theoretical peak memory bandwidth
 */
inline float getTheoreticalPeakBandwidth() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    // Approximate bandwidth based on bus width
    // Modern GDDR6 typically runs at ~14-19 Gbps per pin
    float memorySpeedGbps = 14.0f;  // Conservative GDDR6 estimate
    float peakBandwidth = memorySpeedGbps * (prop.memoryBusWidth / 8);
    
    return peakBandwidth;
}
