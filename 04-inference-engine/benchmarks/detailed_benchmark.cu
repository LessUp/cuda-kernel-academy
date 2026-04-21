#include "common.h"
#include "kernels.cuh"
#include "vectorized_gemm.cuh"
#include "profiler.h"
#include "autotuner.h"
#include <iostream>
#include <iomanip>

using namespace mini_inference;

void run_detailed_benchmark(int M, int N, int K, cublasHandle_t cublas_handle) {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Detailed Benchmark: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════╝\n";

    // Allocate memory
    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));

    std::vector<float> h_A(M * K), h_B(K * N);
    random_init(h_A.data(), M * K);
    random_init(h_B.data(), K * N);
    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));

    Profiler profiler;
    std::vector<ProfileResult> results;

    // Profile each kernel
    auto profile_kernel = [&](const std::string& name, auto kernel_func) {
        auto result = profiler.profile(name, M, N, K, 10, 50, kernel_func);
        results.push_back(result);
        return result;
    };

    // cuBLAS (baseline)
    profile_kernel("cuBLAS", [&]() {
        launch_cublas_gemm(cublas_handle, d_A.get(), d_B.get(), d_C.get(), M, N, K);
    });

    // Naive
    profile_kernel("Naive", [&]() {
        launch_naive_matmul(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    });

    // Tiled
    profile_kernel("Tiled", [&]() {
        launch_tiled_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    });

    // Coalesced
    profile_kernel("Coalesced", [&]() {
        launch_coalesced_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    });

    // Double Buffer
    profile_kernel("DoubleBuffer", [&]() {
        launch_double_buffer_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    });

    // Optimized
    profile_kernel("Optimized", [&]() {
        launch_optimized_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    });

    // Vectorized
    profile_kernel("Vectorized", [&]() {
        launch_vectorized_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    });

    // Fused
    profile_kernel("Fused", [&]() {
        launch_fused_gemm(d_A.get(), d_B.get(), d_C.get(), nullptr, M, N, K, false, false);
    });

    // Print comparison
    Profiler::print_comparison(results, "cuBLAS");

    // Roofline analysis
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // Estimate peak performance (rough)
    // Note: clockRate and memoryClockRate are deprecated in newer CUDA
    // Using conservative estimates based on SM count
    float peak_gflops = prop.multiProcessorCount *
                        (prop.major >= 8 ? 256 : 128) *  // FP32 cores per SM
                        1.5f * 2;     // Assume ~1.5 GHz, FMA = 2 FLOPs
    float peak_bandwidth = prop.memoryBusWidth / 8.0f *
                           10.0f * 2;  // Assume ~10 Gbps per pin, DDR

    RooflineAnalyzer roofline(peak_gflops, peak_bandwidth);
    for (const auto& r : results) {
        roofline.add_point(r);
    }
    roofline.analyze();
}

void run_scaling_benchmark(cublasHandle_t cublas_handle) {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                           Scaling Benchmark                                   ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════╝\n";

    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};

    std::cout << "\n";
    std::cout << std::setw(8) << "Size" << " │ "
              << std::setw(12) << "cuBLAS" << " │ "
              << std::setw(12) << "Optimized" << " │ "
              << std::setw(12) << "Vectorized" << " │ "
              << std::setw(10) << "Ratio" << "\n";
    std::cout << "─────────┼──────────────┼──────────────┼──────────────┼────────────\n";

    for (int size : sizes) {
        DeviceMemory d_A(size * size * sizeof(float));
        DeviceMemory d_B(size * size * sizeof(float));
        DeviceMemory d_C(size * size * sizeof(float));

        std::vector<float> h_A(size * size), h_B(size * size);
        random_init(h_A.data(), size * size);
        random_init(h_B.data(), size * size);
        d_A.copy_from_host(h_A.data(), size * size * sizeof(float));
        d_B.copy_from_host(h_B.data(), size * size * sizeof(float));

        Profiler profiler;

        auto cublas_result = profiler.profile("cuBLAS", size, size, size, 5, 20, [&]() {
            launch_cublas_gemm(cublas_handle, d_A.get(), d_B.get(), d_C.get(), size, size, size);
        });

        auto opt_result = profiler.profile("Optimized", size, size, size, 5, 20, [&]() {
            launch_optimized_gemm(d_A.get(), d_B.get(), d_C.get(), size, size, size);
        });

        auto vec_result = profiler.profile("Vectorized", size, size, size, 5, 20, [&]() {
            launch_vectorized_gemm(d_A.get(), d_B.get(), d_C.get(), size, size, size);
        });

        float best_ratio = std::max(opt_result.gflops, vec_result.gflops) / cublas_result.gflops * 100;

        std::cout << std::setw(8) << size << " │ "
                  << std::setw(10) << std::fixed << std::setprecision(1) << cublas_result.gflops << " │ "
                  << std::setw(10) << opt_result.gflops << " │ "
                  << std::setw(10) << vec_result.gflops << " │ "
                  << std::setw(8) << std::setprecision(1) << best_ratio << "%\n";
    }
}

void run_autotuner_demo(cublasHandle_t cublas_handle) {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                           Auto-Tuner Demo                                     ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════╝\n";

    AutoTuner tuner;

    std::vector<std::tuple<int, int, int>> test_sizes = {
        {256, 256, 256},
        {512, 512, 256},
        {1024, 1024, 512},
        {2048, 2048, 1024}
    };

    for (const auto& [M, N, K] : test_sizes) {
        std::cout << "\nTuning for " << M << "x" << N << "x" << K << "...\n";
        auto result = tuner.tune(M, N, K, cublas_handle);
        std::cout << "  Best kernel: " << kernel_type_name(result.config.kernel_type) << "\n";
        std::cout << "  Performance: " << std::fixed << std::setprecision(2)
                  << result.gflops << " GFLOPS\n";
    }

    tuner.print_cache();
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        // Print GPU info
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              Mini-Inference Engine - Detailed Performance Analysis            ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";
        std::cout << "GPU: " << prop.name << "\n";
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "Memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB\n";
        std::cout << "SM Count: " << prop.multiProcessorCount << "\n";
        std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits\n";

        cublasHandle_t cublas_handle;
        CUBLAS_CHECK(cublasCreate(&cublas_handle));

        // Run benchmarks
        run_detailed_benchmark(1024, 1024, 512, cublas_handle);
        run_scaling_benchmark(cublas_handle);
        run_autotuner_demo(cublas_handle);

        CUBLAS_CHECK(cublasDestroy(cublas_handle));

        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                           Benchmark Complete                                  ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════════════════════╝\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
