/**
 * @file test_softmax.cpp
 * @brief Tests for softmax kernel
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>

#include "tensorcraft/core/cuda_check.hpp"
#include "tensorcraft/kernels/softmax.hpp"

using namespace tensorcraft;
using namespace tensorcraft::kernels;

class SoftmaxTest : public ::testing::Test {
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
    
    std::vector<float> reference_softmax(const std::vector<float>& input, int rows, int cols) {
        std::vector<float> output(input.size());
        
        for (int r = 0; r < rows; ++r) {
            // Find max
            float max_val = input[r * cols];
            for (int c = 1; c < cols; ++c) {
                max_val = std::max(max_val, input[r * cols + c]);
            }
            
            // Compute exp and sum
            float sum = 0.0f;
            for (int c = 0; c < cols; ++c) {
                output[r * cols + c] = std::exp(input[r * cols + c] - max_val);
                sum += output[r * cols + c];
            }
            
            // Normalize
            for (int c = 0; c < cols; ++c) {
                output[r * cols + c] /= sum;
            }
        }
        
        return output;
    }
    
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
};

TEST_F(SoftmaxTest, RowSumInvariant) {
    // Property 6: Softmax Row Sum Invariant
    const int rows = 100;
    const int cols = 512;
    
    auto h_input = random_vector(rows * cols);
    std::vector<float> h_output(rows * cols);
    
    float *d_input, *d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_input, rows * cols * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_output, rows * cols * sizeof(float)));
    
    TC_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    
    softmax(d_input, d_output, rows, cols);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    TC_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Check row sums equal 1
    for (int r = 0; r < rows; ++r) {
        float row_sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            row_sum += h_output[r * cols + c];
            EXPECT_GE(h_output[r * cols + c], 0.0f);  // All values >= 0
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5f);
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(SoftmaxTest, NumericalCorrectness) {
    const int rows = 50;
    const int cols = 256;
    
    auto h_input = random_vector(rows * cols);
    std::vector<float> h_output(rows * cols);
    auto h_ref = reference_softmax(h_input, rows, cols);
    
    float *d_input, *d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_input, rows * cols * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_output, rows * cols * sizeof(float)));
    
    TC_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    
    softmax(d_input, d_output, rows, cols);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    TC_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (size_t i = 0; i < h_output.size(); ++i) {
        EXPECT_NEAR(h_output[i], h_ref[i], 1e-5f);
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(SoftmaxTest, LargeValues) {
    // Test numerical stability with large values
    const int rows = 10;
    const int cols = 100;
    
    std::vector<float> h_input(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        h_input[i] = 100.0f + dist(gen);  // Large values
    }
    std::vector<float> h_output(rows * cols);
    
    float *d_input, *d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_input, rows * cols * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_output, rows * cols * sizeof(float)));
    
    TC_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    
    softmax(d_input, d_output, rows, cols);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    TC_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Should still sum to 1 and have no NaN/Inf
    for (int r = 0; r < rows; ++r) {
        float row_sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float val = h_output[r * cols + c];
            EXPECT_FALSE(std::isnan(val));
            EXPECT_FALSE(std::isinf(val));
            row_sum += val;
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-4f);
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
}
