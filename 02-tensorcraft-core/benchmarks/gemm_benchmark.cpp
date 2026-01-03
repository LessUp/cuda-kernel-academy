/**
 * @file gemm_benchmark.cpp
 * @brief GEMM performance benchmarks
 */

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>

#include "tensorcraft/core/cuda_check.hpp"
#include "tensorcraft/kernels/gemm.hpp"

using namespace tensorcraft;
using namespace tensorcraft::kernels;

class GemmBenchmark : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        int size = state.range(0);
        M = N = K = size;
        
        // Allocate device memory
        TC_CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        TC_CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        TC_CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        
        // Initialize with random data
        std::vector<float> h_A(M * K), h_B(K * N);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& x : h_A) x = dist(gen);
        for (auto& x : h_B) x = dist(gen);
        
        TC_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        TC_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
        TC_CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
    }
    
    void TearDown(const benchmark::State&) override {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    
protected:
    float *d_A, *d_B, *d_C;
    int M, N, K;
};

BENCHMARK_DEFINE_F(GemmBenchmark, Naive)(benchmark::State& state) {
    for (auto _ : state) {
        launch_gemm(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, GemmVersion::Naive);
        cudaDeviceSynchronize();
    }
    
    // Report GFLOPS
    double gflops = 2.0 * M * N * K / 1e9;
    state.counters["GFLOPS"] = benchmark::Counter(
        gflops, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK_DEFINE_F(GemmBenchmark, Tiled)(benchmark::State& state) {
    for (auto _ : state) {
        launch_gemm(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, GemmVersion::Tiled);
        cudaDeviceSynchronize();
    }
    
    double gflops = 2.0 * M * N * K / 1e9;
    state.counters["GFLOPS"] = benchmark::Counter(
        gflops, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK_DEFINE_F(GemmBenchmark, DoubleBuffer)(benchmark::State& state) {
    for (auto _ : state) {
        launch_gemm(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, GemmVersion::DoubleBuffer);
        cudaDeviceSynchronize();
    }
    
    double gflops = 2.0 * M * N * K / 1e9;
    state.counters["GFLOPS"] = benchmark::Counter(
        gflops, benchmark::Counter::kIsIterationInvariantRate);
}

// Register benchmarks with different matrix sizes
BENCHMARK_REGISTER_F(GemmBenchmark, Naive)
    ->RangeMultiplier(2)
    ->Range(256, 2048)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(GemmBenchmark, Tiled)
    ->RangeMultiplier(2)
    ->Range(256, 2048)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(GemmBenchmark, DoubleBuffer)
    ->RangeMultiplier(2)
    ->Range(256, 2048)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
