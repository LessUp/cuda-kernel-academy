/**
 * @file test_elementwise.cpp
 * @brief Tests for elementwise kernels
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>

#include "tensorcraft/core/cuda_check.hpp"
#include "tensorcraft/kernels/elementwise.hpp"

using namespace tensorcraft;
using namespace tensorcraft::kernels;

class ElementwiseTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        gen = std::mt19937(rd());
        dist = std::uniform_real_distribution<float>(-10.0f, 10.0f);
    }
    
    std::vector<float> random_vector(size_t n) {
        std::vector<float> v(n);
        for (auto& x : v) x = dist(gen);
        return v;
    }
    
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
};

TEST_F(ElementwiseTest, VectorAddCorrectness) {
    const size_t n = 10000;
    auto h_a = random_vector(n);
    auto h_b = random_vector(n);
    std::vector<float> h_c(n);
    
    float *d_a, *d_b, *d_c;
    TC_CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));
    
    TC_CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    vector_add(d_a, d_b, d_c, n);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    TC_CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(h_c[i], h_a[i] + h_b[i], 1e-5f);
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

TEST_F(ElementwiseTest, ReLUCorrectness) {
    const size_t n = 10000;
    auto h_input = random_vector(n);
    std::vector<float> h_output(n);
    
    float *d_input, *d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    
    TC_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    relu(d_input, d_output, n);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    TC_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (size_t i = 0; i < n; ++i) {
        float expected = std::max(0.0f, h_input[i]);
        EXPECT_NEAR(h_output[i], expected, 1e-5f);
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(ElementwiseTest, SiLUCorrectness) {
    const size_t n = 10000;
    auto h_input = random_vector(n);
    std::vector<float> h_output(n);
    
    float *d_input, *d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    
    TC_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    silu(d_input, d_output, n);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    TC_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (size_t i = 0; i < n; ++i) {
        float x = h_input[i];
        float expected = x / (1.0f + std::exp(-x));
        EXPECT_NEAR(h_output[i], expected, 1e-4f);
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(ElementwiseTest, GeLUCorrectness) {
    const size_t n = 10000;
    auto h_input = random_vector(n);
    std::vector<float> h_output(n);
    
    float *d_input, *d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    
    TC_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    gelu(d_input, d_output, n);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    TC_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    
    for (size_t i = 0; i < n; ++i) {
        float x = h_input[i];
        float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        float expected = 0.5f * x * (1.0f + std::tanh(inner));
        EXPECT_NEAR(h_output[i], expected, 1e-4f);
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(ElementwiseTest, EmptyInput) {
    float *d_input = nullptr, *d_output = nullptr;
    
    // Should not crash with empty input
    relu(d_input, d_output, 0);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
}
