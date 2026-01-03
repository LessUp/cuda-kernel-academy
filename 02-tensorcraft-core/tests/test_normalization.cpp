/**
 * @file test_normalization.cpp
 * @brief Tests for normalization kernels (LayerNorm, RMSNorm, BatchNorm)
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>

#include "tensorcraft/core/cuda_check.hpp"
#include "tensorcraft/kernels/normalization.hpp"

using namespace tensorcraft;
using namespace tensorcraft::kernels;

class NormalizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        gen = std::mt19937(rd());
        dist = std::uniform_real_distribution<float>(-5.0f, 5.0f);
    }
    
    std::vector<float> random_vector(size_t n) {
        std::vector<float> v(n);
        for (auto& x : v) x = dist(gen);
        return v;
    }
    
    std::vector<float> ones(size_t n) {
        return std::vector<float>(n, 1.0f);
    }
    
    std::vector<float> zeros(size_t n) {
        return std::vector<float>(n, 0.0f);
    }
    
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
};

TEST_F(NormalizationTest, LayerNormStatisticalProperties) {
    // Property 7: LayerNorm output should have mean ≈ 0 and var ≈ 1
    const int batch_size = 32;
    const int hidden_size = 256;
    const float eps = 1e-5f;
    
    auto h_input = random_vector(batch_size * hidden_size);
    auto h_gamma = ones(hidden_size);
    auto h_beta = zeros(hidden_size);
    std::vector<float> h_output(batch_size * hidden_size);
    
    float *d_input, *d_gamma, *d_beta, *d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_input, batch_size * hidden_size * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_gamma, hidden_size * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_beta, hidden_size * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_output, batch_size * hidden_size * sizeof(float)));
    
    TC_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), batch_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    layernorm(d_input, d_gamma, d_beta, d_output, batch_size, hidden_size, eps);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    TC_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Check each row has mean ≈ 0 and var ≈ 1
    for (int b = 0; b < batch_size; ++b) {
        float mean = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            mean += h_output[b * hidden_size + h];
        }
        mean /= hidden_size;
        
        float var = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            float diff = h_output[b * hidden_size + h] - mean;
            var += diff * diff;
        }
        var /= hidden_size;
        
        EXPECT_NEAR(mean, 0.0f, 1e-4f);
        EXPECT_NEAR(var, 1.0f, 1e-3f);
    }
    
    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_output);
}

TEST_F(NormalizationTest, LayerNormWithAffine) {
    const int batch_size = 16;
    const int hidden_size = 128;
    const float eps = 1e-5f;
    
    auto h_input = random_vector(batch_size * hidden_size);
    auto h_gamma = random_vector(hidden_size);
    auto h_beta = random_vector(hidden_size);
    std::vector<float> h_output(batch_size * hidden_size);
    
    float *d_input, *d_gamma, *d_beta, *d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_input, batch_size * hidden_size * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_gamma, hidden_size * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_beta, hidden_size * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_output, batch_size * hidden_size * sizeof(float)));
    
    TC_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), batch_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    layernorm(d_input, d_gamma, d_beta, d_output, batch_size, hidden_size, eps);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    TC_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify against reference implementation
    for (int b = 0; b < batch_size; ++b) {
        // Compute mean
        float mean = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            mean += h_input[b * hidden_size + h];
        }
        mean /= hidden_size;
        
        // Compute variance
        float var = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            float diff = h_input[b * hidden_size + h] - mean;
            var += diff * diff;
        }
        var /= hidden_size;
        
        float inv_std = 1.0f / std::sqrt(var + eps);
        
        // Check output
        for (int h = 0; h < hidden_size; ++h) {
            float normalized = (h_input[b * hidden_size + h] - mean) * inv_std;
            float expected = normalized * h_gamma[h] + h_beta[h];
            EXPECT_NEAR(h_output[b * hidden_size + h], expected, 1e-4f);
        }
    }
    
    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_output);
}

TEST_F(NormalizationTest, RMSNormCorrectness) {
    // Property 8: RMSNorm Correctness
    const int batch_size = 32;
    const int hidden_size = 256;
    const float eps = 1e-6f;
    
    auto h_input = random_vector(batch_size * hidden_size);
    auto h_weight = random_vector(hidden_size);
    std::vector<float> h_output(batch_size * hidden_size);
    
    float *d_input, *d_weight, *d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_input, batch_size * hidden_size * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_weight, hidden_size * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_output, batch_size * hidden_size * sizeof(float)));
    
    TC_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), batch_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    rmsnorm(d_input, d_weight, d_output, batch_size, hidden_size, eps);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    TC_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify against reference
    for (int b = 0; b < batch_size; ++b) {
        // Compute RMS
        float sum_sq = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            float val = h_input[b * hidden_size + h];
            sum_sq += val * val;
        }
        float rms_inv = 1.0f / std::sqrt(sum_sq / hidden_size + eps);
        
        // Check output
        for (int h = 0; h < hidden_size; ++h) {
            float expected = h_input[b * hidden_size + h] * rms_inv * h_weight[h];
            EXPECT_NEAR(h_output[b * hidden_size + h], expected, 1e-4f);
        }
    }
    
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

TEST_F(NormalizationTest, BatchNormInference) {
    const int N = 4, C = 8, H = 16, W = 16;
    const float eps = 1e-5f;
    
    auto h_input = random_vector(N * C * H * W);
    auto h_gamma = random_vector(C);
    auto h_beta = random_vector(C);
    std::vector<float> h_running_mean(C);
    std::vector<float> h_running_var(C);
    std::vector<float> h_output(N * C * H * W);
    
    // Initialize running stats
    for (int c = 0; c < C; ++c) {
        h_running_mean[c] = dist(gen) * 0.1f;
        h_running_var[c] = std::abs(dist(gen)) + 0.1f;  // Positive variance
    }
    
    float *d_input, *d_gamma, *d_beta, *d_mean, *d_var, *d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_input, N * C * H * W * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_gamma, C * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_beta, C * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_mean, C * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_var, C * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_output, N * C * H * W * sizeof(float)));
    
    TC_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), C * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), C * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_mean, h_running_mean.data(), C * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_var, h_running_var.data(), C * sizeof(float), cudaMemcpyHostToDevice));
    
    launch_batchnorm(d_input, d_gamma, d_beta, d_mean, d_var, d_output, N, C, H, W, eps, false);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    TC_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * C * H * W * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify against reference
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float inv_std = 1.0f / std::sqrt(h_running_var[c] + eps);
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int idx = ((n * C + c) * H + h) * W + w;
                    float normalized = (h_input[idx] - h_running_mean[c]) * inv_std;
                    float expected = normalized * h_gamma[c] + h_beta[c];
                    EXPECT_NEAR(h_output[idx], expected, 1e-4f);
                }
            }
        }
    }
    
    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_mean);
    cudaFree(d_var);
    cudaFree(d_output);
}
