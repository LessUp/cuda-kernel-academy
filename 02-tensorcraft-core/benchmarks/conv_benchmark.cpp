/**
 * @file conv_benchmark.cpp
 * @brief Convolution kernel performance benchmarks
 */

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>

#include "tensorcraft/core/cuda_check.hpp"
#include "tensorcraft/kernels/conv2d.hpp"

using namespace tensorcraft;
using namespace tensorcraft::kernels;

class Conv2DBenchmark : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        N = 1;
        C = state.range(0);
        H = W = state.range(1);
        K = state.range(2);
        R = S = 3;
        stride = 1;
        padding = 1;
        
        OH = (H + 2 * padding - R) / stride + 1;
        OW = (W + 2 * padding - S) / stride + 1;
        
        TC_CUDA_CHECK(cudaMalloc(&d_input, N * C * H * W * sizeof(float)));
        TC_CUDA_CHECK(cudaMalloc(&d_weight, K * C * R * S * sizeof(float)));
        TC_CUDA_CHECK(cudaMalloc(&d_output, N * K * OH * OW * sizeof(float)));
        
        std::vector<float> h_input(N * C * H * W);
        std::vector<float> h_weight(K * C * R * S);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& x : h_input) x = dist(gen);
        for (auto& x : h_weight) x = dist(gen);
        
        TC_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
        TC_CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), K * C * R * S * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    void TearDown(const benchmark::State&) override {
        cudaFree(d_input);
        cudaFree(d_weight);
        cudaFree(d_output);
    }
    
protected:
    float *d_input, *d_weight, *d_output;
    int N, C, H, W, K, R, S, OH, OW;
    int stride, padding;
};

BENCHMARK_DEFINE_F(Conv2DBenchmark, Naive)(benchmark::State& state) {
    for (auto _ : state) {
        launch_conv2d_naive(d_input, d_weight, nullptr, d_output,
                           N, C, H, W, K, R, S, stride, stride, padding, padding);
        cudaDeviceSynchronize();
    }
    
    // Report GFLOPS: 2 * N * K * OH * OW * C * R * S
    double gflops = 2.0 * N * K * OH * OW * C * R * S / 1e9;
    state.counters["GFLOPS"] = benchmark::Counter(
        gflops, benchmark::Counter::kIsIterationInvariantRate);
}

// Conv2D benchmarks: (C, H/W, K)
BENCHMARK_REGISTER_F(Conv2DBenchmark, Naive)
    ->Args({64, 56, 64})    // ResNet-like
    ->Args({128, 28, 128})
    ->Args({256, 14, 256})
    ->Args({512, 7, 512})
    ->Args({64, 224, 64})   // First layer
    ->Unit(benchmark::kMillisecond);

// Depthwise Conv2D Benchmark
class DepthwiseConv2DBenchmark : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        N = 1;
        C = state.range(0);
        H = W = state.range(1);
        R = S = 3;
        stride = 1;
        padding = 1;
        
        OH = (H + 2 * padding - R) / stride + 1;
        OW = (W + 2 * padding - S) / stride + 1;
        
        TC_CUDA_CHECK(cudaMalloc(&d_input, N * C * H * W * sizeof(float)));
        TC_CUDA_CHECK(cudaMalloc(&d_weight, C * R * S * sizeof(float)));
        TC_CUDA_CHECK(cudaMalloc(&d_output, N * C * OH * OW * sizeof(float)));
        
        std::vector<float> h_input(N * C * H * W);
        std::vector<float> h_weight(C * R * S);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& x : h_input) x = dist(gen);
        for (auto& x : h_weight) x = dist(gen);
        
        TC_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
        TC_CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), C * R * S * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    void TearDown(const benchmark::State&) override {
        cudaFree(d_input);
        cudaFree(d_weight);
        cudaFree(d_output);
    }
    
protected:
    float *d_input, *d_weight, *d_output;
    int N, C, H, W, R, S, OH, OW;
    int stride, padding;
};

BENCHMARK_DEFINE_F(DepthwiseConv2DBenchmark, Depthwise)(benchmark::State& state) {
    for (auto _ : state) {
        launch_conv2d_depthwise(d_input, d_weight, nullptr, d_output,
                               N, C, H, W, R, S, stride, stride, padding, padding);
        cudaDeviceSynchronize();
    }
    
    double gflops = 2.0 * N * C * OH * OW * R * S / 1e9;
    state.counters["GFLOPS"] = benchmark::Counter(
        gflops, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK_REGISTER_F(DepthwiseConv2DBenchmark, Depthwise)
    ->Args({32, 112})
    ->Args({64, 56})
    ->Args({128, 28})
    ->Args({256, 14})
    ->Args({512, 7})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
