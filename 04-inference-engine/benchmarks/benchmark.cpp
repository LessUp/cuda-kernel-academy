#include <iomanip>
#include <iostream>
#include <vector>

#include "common.h"
#include "kernels.cuh"

using namespace mini_inference;

void print_header() {
    std::cout << "\n";
    std::cout
        << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout
        << "║                    Mini-Inference Engine GEMM Benchmark                      ║\n";
    std::cout
        << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

void print_results(const std::string& name, const PerfStats& stats) {
    std::cout << std::setw(16) << name << " | " << std::setw(8) << std::fixed
              << std::setprecision(3) << stats.kernel_time_ms << " ms"
              << " | " << std::setw(10) << std::fixed << std::setprecision(2) << stats.gflops
              << " GFLOPS"
              << " | " << std::setw(8) << std::fixed << std::setprecision(1)
              << stats.cublas_ratio * 100 << "%"
              << "\n";
}

void run_benchmark(int M, int N, int K, cublasHandle_t cublas_handle) {
    std::cout
        << "\n┌──────────────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│ Matrix Size: M=" << M << ", N=" << N << ", K=" << K << " (" << std::fixed
              << std::setprecision(2) << (2.0 * M * N * K / 1e9) << " GFLOPs)\n";
    std::cout
        << "├──────────────────────────────────────────────────────────────────────────────┤\n";
    std::cout << std::setw(16) << "Kernel"
              << " | " << std::setw(12) << "Time"
              << " | " << std::setw(14) << "Performance"
              << " | " << std::setw(10) << "vs cuBLAS"
              << "\n";
    std::cout
        << "├──────────────────────────────────────────────────────────────────────────────┤\n";

    // Allocate memory
    std::vector<float> h_A(M * K), h_B(K * N);
    random_init(h_A.data(), M * K);
    random_init(h_B.data(), K * N);

    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));

    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));

    // Benchmark cuBLAS first to get reference
    PerfStats cublas_stats = benchmark_kernel(GemmKernelType::CUBLAS, d_A.get(), d_B.get(),
                                              d_C.get(), M, N, K, 10, 50, cublas_handle);
    cublas_stats.cublas_ratio = 1.0f;
    print_results("cuBLAS", cublas_stats);

    // Benchmark each kernel
    std::vector<std::pair<GemmKernelType, std::string>> kernels = {
        {GemmKernelType::NAIVE, "Naive"},
        {GemmKernelType::TILED, "Tiled"},
        {GemmKernelType::COALESCED, "Coalesced"},
        {GemmKernelType::DOUBLE_BUFFER, "DoubleBuffer"},
        {GemmKernelType::REGISTER_BLOCKED, "Optimized"},
        {GemmKernelType::FUSED, "Fused"}};

    for (const auto& [type, name] : kernels) {
        PerfStats stats =
            benchmark_kernel(type, d_A.get(), d_B.get(), d_C.get(), M, N, K, 10, 50, cublas_handle);
        stats.cublas_ratio = stats.gflops / cublas_stats.gflops;
        print_results(name, stats);
    }

    std::cout
        << "└──────────────────────────────────────────────────────────────────────────────┘\n";
}

void verify_correctness(int M, int N, int K) {
    std::cout
        << "\n┌──────────────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│ Correctness Verification: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout
        << "├──────────────────────────────────────────────────────────────────────────────┤\n";

    std::vector<float> h_A(M * K), h_B(K * N), h_C_ref(M * N);
    random_init(h_A.data(), M * K);
    random_init(h_B.data(), K * N);

    // CPU reference
    cpu_matmul(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));

    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));

    std::vector<std::pair<GemmKernelType, std::string>> kernels = {
        {GemmKernelType::NAIVE, "Naive"},
        {GemmKernelType::TILED, "Tiled"},
        {GemmKernelType::COALESCED, "Coalesced"},
        {GemmKernelType::DOUBLE_BUFFER, "DoubleBuffer"},
        {GemmKernelType::REGISTER_BLOCKED, "Optimized"},
        {GemmKernelType::FUSED, "Fused"}};

    std::vector<float> h_C(M * N);

    for (const auto& [type, name] : kernels) {
        d_C.zero();

        switch (type) {
            case GemmKernelType::NAIVE:
                launch_naive_matmul(d_A.get(), d_B.get(), d_C.get(), M, N, K);
                break;
            case GemmKernelType::TILED:
                launch_tiled_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
                break;
            case GemmKernelType::COALESCED:
                launch_coalesced_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
                break;
            case GemmKernelType::DOUBLE_BUFFER:
                launch_double_buffer_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
                break;
            case GemmKernelType::REGISTER_BLOCKED:
                launch_optimized_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
                break;
            case GemmKernelType::FUSED:
                launch_fused_gemm(d_A.get(), d_B.get(), d_C.get(), nullptr, M, N, K, false, false);
                break;
            default:
                break;
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        d_C.copy_to_host(h_C.data(), M * N * sizeof(float));

        float max_error = compare_matrices(h_C.data(), h_C_ref.data(), M * N);
        std::cout << "│ " << std::setw(14) << name << ": max error = " << std::scientific
                  << std::setprecision(2) << max_error << (max_error < 1e-4f ? " ✓" : " ✗") << "\n";
    }

    std::cout
        << "└──────────────────────────────────────────────────────────────────────────────┘\n";
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    try {
        CUDA_CHECK(cudaSetDevice(0));

        // Print GPU info
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

        print_header();
        std::cout << "GPU: " << prop.name << "\n";
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "Memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB\n";
        std::cout << "SM Count: " << prop.multiProcessorCount << "\n";

        // Create cuBLAS handle
        cublasHandle_t cublas_handle;
        CUBLAS_CHECK(cublasCreate(&cublas_handle));

        // Verify correctness first
        verify_correctness(256, 256, 128);

        // Run benchmarks for different sizes
        std::vector<int> sizes = {256, 512, 1024, 2048, 4096};

        for (int size : sizes) {
            run_benchmark(size, size, size, cublas_handle);
        }

        // Summary
        std::cout << "\n";
        std::cout
            << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout
            << "║                              Benchmark Complete                              ║\n";
        std::cout
            << "╚══════════════════════════════════════════════════════════════════════════════╝\n";

        CUBLAS_CHECK(cublasDestroy(cublas_handle));

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
