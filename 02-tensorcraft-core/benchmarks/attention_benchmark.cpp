/**
 * @file attention_benchmark.cpp
 * @brief Attention kernel performance benchmarks
 */

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>

#include "tensorcraft/core/cuda_check.hpp"
#include "tensorcraft/kernels/attention.hpp"
#include "tensorcraft/kernels/softmax.hpp"

using namespace tensorcraft;
using namespace tensorcraft::kernels;

class SoftmaxBenchmark : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        rows = state.range(0);
        cols = state.range(1);
        
        TC_CUDA_CHECK(cudaMalloc(&d_input, rows * cols * sizeof(float)));
        TC_CUDA_CHECK(cudaMalloc(&d_output, rows * cols * sizeof(float)));
        
        std::vector<float> h_input(rows * cols);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for (auto& x : h_input) x = dist(gen);
        
        TC_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    void TearDown(const benchmark::State&) override {
        cudaFree(d_input);
        cudaFree(d_output);
    }
    
protected:
    float *d_input, *d_output;
    int rows, cols;
};

BENCHMARK_DEFINE_F(SoftmaxBenchmark, Softmax)(benchmark::State& state) {
    for (auto _ : state) {
        softmax(d_input, d_output, rows, cols);
        cudaDeviceSynchronize();
    }
    
    // Report bandwidth
    double bytes = 2.0 * rows * cols * sizeof(float);  // Read + write
    state.counters["Bandwidth"] = benchmark::Counter(
        bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1024);
}

// Softmax benchmarks: (rows, cols)
BENCHMARK_REGISTER_F(SoftmaxBenchmark, Softmax)
    ->Args({1024, 512})
    ->Args({1024, 1024})
    ->Args({1024, 2048})
    ->Args({1024, 4096})
    ->Args({4096, 1024})
    ->Args({4096, 4096})
    ->Unit(benchmark::kMicrosecond);

// RoPE Benchmark
class RoPEBenchmark : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        batch_size = 1;
        seq_len = state.range(0);
        num_heads = 32;
        head_dim = 128;
        
        size_t x_size = batch_size * seq_len * num_heads * head_dim;
        size_t cache_size = seq_len * (head_dim / 2);
        
        TC_CUDA_CHECK(cudaMalloc(&d_x, x_size * sizeof(float)));
        TC_CUDA_CHECK(cudaMalloc(&d_cos, cache_size * sizeof(float)));
        TC_CUDA_CHECK(cudaMalloc(&d_sin, cache_size * sizeof(float)));
        
        // Initialize
        std::vector<float> h_x(x_size);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& x : h_x) x = dist(gen);
        
        TC_CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Precompute cache
        precompute_rope_cache(d_cos, d_sin, seq_len, head_dim);
        TC_CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    void TearDown(const benchmark::State&) override {
        cudaFree(d_x);
        cudaFree(d_cos);
        cudaFree(d_sin);
    }
    
protected:
    float *d_x, *d_cos, *d_sin;
    int batch_size, seq_len, num_heads, head_dim;
};

BENCHMARK_DEFINE_F(RoPEBenchmark, RoPE)(benchmark::State& state) {
    for (auto _ : state) {
        launch_rope(d_x, d_cos, d_sin, batch_size, seq_len, num_heads, head_dim, 0);
        cudaDeviceSynchronize();
    }
    
    double bytes = 2.0 * batch_size * seq_len * num_heads * head_dim * sizeof(float);
    state.counters["Bandwidth"] = benchmark::Counter(
        bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1024);
}

BENCHMARK_REGISTER_F(RoPEBenchmark, RoPE)
    ->Arg(128)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(4096)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
