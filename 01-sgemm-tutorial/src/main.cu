/**
 * SGEMM Optimization Benchmark
 * 
 * This program benchmarks various SGEMM implementations from naive to
 * Tensor Core optimized, demonstrating the performance evolution of
 * GPU matrix multiplication optimization techniques.
 * 
 * Implementations:
 * 1. Naive - Basic three-loop implementation
 * 2. Tiled - Shared memory tiling
 * 3. Bank Conflict Free - Tiled with padding to avoid bank conflicts
 * 4. Double Buffer - Software pipelining with double buffering
 * 5. Tensor Core - WMMA API for Tensor Core acceleration
 * 6. cuBLAS - Reference implementation (performance ceiling)
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#include "utils/cuda_utils.cuh"
#include "utils/benchmark.cuh"
#include "utils/verify.cuh"

#include "kernels/naive_sgemm.cuh"
#include "kernels/tiled_sgemm.cuh"
#include "kernels/bank_conflict_free_sgemm.cuh"
#include "kernels/double_buffer_sgemm.cuh"
#include "kernels/tensor_core_sgemm.cuh"

// ============================================================================
// Configuration
// ============================================================================

// Default matrix sizes to benchmark
std::vector<int> DEFAULT_SIZES = {512, 1024, 2048, 4096};

// Benchmark configuration
constexpr int WARMUP_RUNS = 5;
constexpr int BENCHMARK_RUNS = 20;

// ============================================================================
// Kernel Wrappers
// ============================================================================

// Wrapper functions to match the expected signature for benchmarking
void naive_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    launch_naive_sgemm<32>(A, B, C, M, K, N);
}

void tiled_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    launch_tiled_sgemm<32>(A, B, C, M, K, N);
}

void bank_conflict_free_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    launch_bank_conflict_free_sgemm<32>(A, B, C, M, K, N);
}

void double_buffer_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    launch_double_buffer_sgemm<32>(A, B, C, M, K, N);
}

void tensor_core_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    launch_tensor_core_sgemm(A, B, C, M, K, N);
}

// ============================================================================
// Main Benchmark Function
// ============================================================================

void runBenchmarks(int size) {
    int M = size, K = size, N = size;
    
    printf("\n");
    printf("================================================================================\n");
    printf("                    Benchmarking %d x %d x %d SGEMM\n", M, K, N);
    printf("================================================================================\n");
    
    SGEMMBenchmark benchmark;
    
    // Run cuBLAS first as reference
    printf("\nRunning cuBLAS (reference)...\n");
    BenchmarkResult cublas_result = benchmark.runCublas(M, K, N, WARMUP_RUNS, BENCHMARK_RUNS);
    float cublas_gflops = cublas_result.gflops;
    
    // Run each kernel implementation
    printf("Running Naive SGEMM...\n");
    // Use slightly relaxed tolerance to account for floating-point accumulation order differences
    benchmark.run("Naive", naive_kernel, M, K, N, WARMUP_RUNS, BENCHMARK_RUNS, 1e-3f, 1e-4f);
    
    printf("Running Tiled SGEMM...\n");
    benchmark.run("Tiled (32x32)", tiled_kernel, M, K, N, WARMUP_RUNS, BENCHMARK_RUNS, 1e-3f, 1e-4f);
    
    printf("Running Bank Conflict Free SGEMM...\n");
    benchmark.run("Bank Conflict Free", bank_conflict_free_kernel, M, K, N, WARMUP_RUNS, BENCHMARK_RUNS, 1e-3f, 1e-4f);
    
    printf("Running Double Buffer SGEMM...\n");
    benchmark.run("Double Buffer", double_buffer_kernel, M, K, N, WARMUP_RUNS, BENCHMARK_RUNS, 1e-3f, 1e-4f);
    
    // Check if Tensor Cores are available
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (prop.major >= 7) {
        printf("Running Tensor Core SGEMM...\n");
        // Tensor Core uses FP16 intermediate precision, needs relaxed tolerance
        // For K=512, expect ~sqrt(K) * FP16_epsilon ≈ 0.01 error
        benchmark.run("Tensor Core (WMMA)", tensor_core_kernel, M, K, N, 
                      WARMUP_RUNS, BENCHMARK_RUNS, 5e-2f, 1e-2f);
    } else {
        printf("Skipping Tensor Core (requires sm_70+, current: sm_%d%d)\n", 
               prop.major, prop.minor);
    }
    
    // Print results
    benchmark.printSummary();
    
    // Print performance comparison
    printPerformanceComparison(benchmark.getResults(), cublas_gflops);
    
    // Export roofline data
    char filename[256];
    snprintf(filename, sizeof(filename), "roofline_data_%d.csv", size);
    benchmark.exportRooflineData(filename);
}

// ============================================================================
// Print Usage
// ============================================================================

void printUsage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("\nOptions:\n");
    printf("  -s, --size SIZE    Matrix size (default: run all standard sizes)\n");
    printf("  -a, --all          Run all standard sizes (512, 1024, 2048, 4096)\n");
    printf("  -h, --help         Show this help message\n");
    printf("\nExamples:\n");
    printf("  %s -s 1024         Benchmark 1024x1024 matrices\n", program);
    printf("  %s -a              Benchmark all standard sizes\n", program);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    printf("\n");
    printf("================================================================================\n");
    printf("                    SGEMM Optimization Benchmark Suite\n");
    printf("================================================================================\n");
    
    // Print GPU information
    printGPUInfo();
    
    // Print theoretical peaks
    float peakGflops = getTheoreticalPeakGflops();
    float peakBandwidth = getTheoreticalPeakBandwidth();
    printf("Theoretical Peak FP32: %.2f GFLOPS\n", peakGflops);
    printf("Theoretical Peak Bandwidth: %.2f GB/s\n", peakBandwidth);
    printf("\n");
    
    // Parse command line arguments
    std::vector<int> sizes;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-s" || arg == "--size") {
            if (i + 1 < argc) {
                int size = atoi(argv[++i]);
                if (size > 0 && size % 32 == 0) {
                    sizes.push_back(size);
                } else {
                    fprintf(stderr, "Error: Size must be positive and multiple of 32\n");
                    return 1;
                }
            } else {
                fprintf(stderr, "Error: -s requires a size argument\n");
                return 1;
            }
        } else if (arg == "-a" || arg == "--all") {
            sizes = DEFAULT_SIZES;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Default: run 1024x1024 if no size specified
    if (sizes.empty()) {
        sizes.push_back(1024);
    }
    
    // Run benchmarks for each size
    for (int size : sizes) {
        runBenchmarks(size);
    }
    
    printf("\n");
    printf("================================================================================\n");
    printf("                           Benchmark Complete\n");
    printf("================================================================================\n");
    printf("\n");
    printf("Performance Evolution Summary:\n");
    printf("  1. Naive:              Baseline - memory bound, non-coalesced access\n");
    printf("  2. Tiled:              ~2-5x speedup - shared memory reduces global access\n");
    printf("  3. Bank Conflict Free: ~1.1-1.3x over Tiled - eliminates bank conflicts\n");
    printf("  4. Double Buffer:      ~1.1-1.2x over BCF - hides memory latency\n");
    printf("  5. Tensor Core:        ~2-4x over Double Buffer - hardware acceleration\n");
    printf("  6. cuBLAS:             Reference - highly optimized library\n");
    printf("\n");
    printf("Key Optimization Concepts:\n");
    printf("  - Coalescing: Ensure threads access consecutive memory addresses\n");
    printf("  - Tiling: Load data into shared memory for reuse\n");
    printf("  - Bank Conflicts: Avoid multiple threads accessing same memory bank\n");
    printf("  - Latency Hiding: Overlap computation with memory access\n");
    printf("  - Tensor Cores: Use specialized hardware for matrix operations\n");
    printf("\n");
    
    return 0;
}
