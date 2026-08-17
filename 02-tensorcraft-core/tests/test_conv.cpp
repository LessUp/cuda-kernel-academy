/**
 * @file test_conv.cpp
 * @brief Basic correctness tests for 2D convolution kernels
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "tensorcraft/core/cuda_check.hpp"
#include "tensorcraft/kernels/conv2d.hpp"

using namespace tensorcraft::kernels;

namespace {

// CPU reference for the naive conv2d used by the kernel.
std::vector<float> reference_conv2d(const std::vector<float>& input, const std::vector<float>& weight,
                                    const std::vector<float>& bias, int N, int C, int H, int W, int K,
                                    int R, int S, int stride_h, int stride_w, int pad_h, int pad_w) {
    int OH = (H + 2 * pad_h - R) / stride_h + 1;
    int OW = (W + 2 * pad_w - S) / stride_w + 1;
    std::vector<float> output(N * K * OH * OW, 0.0f);

    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    float sum = 0.0f;
                    for (int c = 0; c < C; ++c) {
                        for (int r = 0; r < R; ++r) {
                            for (int s = 0; s < S; ++s) {
                                int ih = oh * stride_h - pad_h + r;
                                int iw = ow * stride_w - pad_w + s;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    sum += input[((n * C + c) * H + ih) * W + iw] *
                                           weight[((k * C + c) * R + r) * S + s];
                                }
                            }
                        }
                    }
                    if (!bias.empty()) {
                        sum += bias[k];
                    }
                    output[((n * K + k) * OH + oh) * OW + ow] = sum;
                }
            }
        }
    }
    return output;
}

}  // namespace

class Conv2DTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices found.";
        }
        TC_CUDA_CHECK(cudaSetDevice(0));
    }

    std::vector<float> random_data(size_t n, float lo = -1.0f, float hi = 1.0f) {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(lo, hi);
        std::vector<float> v(n);
        for (auto& x : v) x = dist(gen);
        return v;
    }

    void run_and_compare(const std::vector<float>& input, const std::vector<float>& weight,
                         const std::vector<float>& bias, int N, int C, int H, int W, int K, int R,
                         int S, int stride_h, int stride_w, int pad_h, int pad_w, float tol = 1e-4f) {
        int OH = (H + 2 * pad_h - R) / stride_h + 1;
        int OW = (W + 2 * pad_w - S) / stride_w + 1;
        const size_t out_size = static_cast<size_t>(N) * K * OH * OW;

        float *d_input, *d_weight, *d_output;
        TC_CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
        TC_CUDA_CHECK(cudaMalloc(&d_weight, weight.size() * sizeof(float)));
        TC_CUDA_CHECK(cudaMalloc(&d_output, out_size * sizeof(float)));

        TC_CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float),
                                 cudaMemcpyHostToDevice));
        TC_CUDA_CHECK(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(float),
                                 cudaMemcpyHostToDevice));

        float* d_bias_ptr = nullptr;
        if (!bias.empty()) {
            TC_CUDA_CHECK(cudaMalloc(&d_bias_ptr, bias.size() * sizeof(float)));
            TC_CUDA_CHECK(cudaMemcpy(d_bias_ptr, bias.data(), bias.size() * sizeof(float),
                                     cudaMemcpyHostToDevice));
        }

        conv2d<float>(d_input, d_weight, d_bias_ptr, d_output, N, C, H, W, K, R, S, stride_h, pad_h);
        TC_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> actual(out_size);
        TC_CUDA_CHECK(cudaMemcpy(actual.data(), d_output, out_size * sizeof(float),
                                 cudaMemcpyDeviceToHost));

        auto expected = reference_conv2d(input, weight, bias, N, C, H, W, K, R, S, stride_h, stride_w,
                                         pad_h, pad_w);
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], tol) << "element " << i;
        }

        cudaFree(d_input);
        cudaFree(d_weight);
        cudaFree(d_output);
        if (d_bias_ptr) cudaFree(d_bias_ptr);
    }
};

// Single channel 4x4 input, 2x2 kernel, no padding/bias.
TEST_F(Conv2DTest, SingleChannelMatchesCpu) {
    const int N = 1, C = 1, H = 4, W = 4, K = 1, R = 2, S = 2;
    auto input = random_data(N * C * H * W);
    auto weight = random_data(K * C * R * S);
    run_and_compare(input, weight, {}, N, C, H, W, K, R, S, 1, 1, 0, 0);
}

// Multi channel 2x3x5x5 input with 3x3 kernel and bias.
TEST_F(Conv2DTest, MultiChannelMatchesCpu) {
    const int N = 2, C = 3, H = 5, W = 5, K = 4, R = 3, S = 3;
    auto input = random_data(N * C * H * W);
    auto weight = random_data(K * C * R * S);
    auto bias = random_data(K, -0.5f, 0.5f);
    run_and_compare(input, weight, bias, N, C, H, W, K, R, S, 1, 1, 1, 1);
}

// padding=1,stride=1 and padding=0,stride=2 configurations.
TEST_F(Conv2DTest, PaddingStrideConfigsMatchCpu) {
    const int N = 1, C = 2, H = 5, W = 5, K = 3, R = 2, S = 2;
    auto input = random_data(N * C * H * W);
    auto weight = random_data(K * C * R * S);
    auto bias = random_data(K, -0.5f, 0.5f);

    // padding=1, stride=1 (same-ish output size)
    run_and_compare(input, weight, bias, N, C, H, W, K, R, S, 1, 1, 1, 1);
    // padding=0, stride=2 (downsampling)
    run_and_compare(input, weight, bias, N, C, H, W, K, R, S, 2, 2, 0, 0);
}
