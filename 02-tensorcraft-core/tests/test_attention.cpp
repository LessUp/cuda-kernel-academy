#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "tensorcraft/kernels/attention.hpp"

using namespace tensorcraft::kernels;

namespace {

std::vector<float> reference_attention(const std::vector<float>& q, const std::vector<float>& k,
                                       const std::vector<float>& v, int batch, int heads, int seq,
                                       int dim, float scale) {
    std::vector<float> out(q.size(), 0.0f);
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            const int base = ((b * heads + h) * seq) * dim;
            for (int m = 0; m < seq; ++m) {
                float max_score = -INFINITY;
                for (int n = 0; n < seq; ++n) {
                    float score = 0.0f;
                    for (int d = 0; d < dim; ++d) {
                        score += q[base + m * dim + d] * k[base + n * dim + d];
                    }
                    max_score = std::fmax(max_score, score * scale);
                }

                float sum_exp = 0.0f;
                std::vector<float> p(seq);
                for (int n = 0; n < seq; ++n) {
                    float score = 0.0f;
                    for (int d = 0; d < dim; ++d) {
                        score += q[base + m * dim + d] * k[base + n * dim + d];
                    }
                    p[n] = std::exp(score * scale - max_score);
                    sum_exp += p[n];
                }

                for (int d = 0; d < dim; ++d) {
                    float acc = 0.0f;
                    for (int n = 0; n < seq; ++n) {
                        acc += (p[n] / sum_exp) * v[base + n * dim + d];
                    }
                    out[base + m * dim + d] = acc;
                }
            }
        }
    }
    return out;
}

}  // namespace

class AttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices found.";
        }
        TC_CUDA_CHECK(cudaSetDevice(0));
    }
};

TEST_F(AttentionTest, FlashAttentionMatchesReference) {
    const int B = 2, H = 3, S = 37, D = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::mt19937 gen(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> q(B * H * S * D), k(B * H * S * D), v(B * H * S * D);
    for (auto& x : q)
        x = dist(gen);
    for (auto& x : k)
        x = dist(gen);
    for (auto& x : v)
        x = dist(gen);

    float *d_q, *d_k, *d_v, *d_o;
    TC_CUDA_CHECK(cudaMalloc(&d_q, q.size() * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_k, k.size() * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_v, v.size() * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_o, q.size() * sizeof(float)));
    TC_CUDA_CHECK(cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_k, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch_flash_attention(d_q, d_k, d_v, d_o, B, H, S, D, scale);
    TC_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual(q.size());
    TC_CUDA_CHECK(cudaMemcpy(actual.data(), d_o, q.size() * sizeof(float), cudaMemcpyDeviceToHost));

    auto expected = reference_attention(q, k, v, B, H, S, D, scale);
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-3f) << "element " << i;
    }

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

TEST_F(AttentionTest, UnsupportedHeadDimThrows) {
    float* dummy = nullptr;
    TC_CUDA_CHECK(cudaMalloc(&dummy, sizeof(float)));
    EXPECT_THROW(launch_flash_attention(dummy, dummy, dummy, dummy, 1, 1, 16, 32, 1.0f),
                 std::invalid_argument);
    TC_CUDA_CHECK(cudaFree(dummy));
}

TEST_F(AttentionTest, NullPointerThrows) {
    EXPECT_THROW(launch_flash_attention(static_cast<float*>(nullptr), static_cast<float*>(nullptr),
                                        static_cast<float*>(nullptr), static_cast<float*>(nullptr),
                                        1, 1, 16, 64, 1.0f),
                 std::invalid_argument);
}

TEST_F(AttentionTest, MoeRouterNumExpertsOutOfRangeThrows) {
    // num_experts > MAX_EXPERTS (8) must fail loudly.
    EXPECT_THROW(launch_moe_router(static_cast<float*>(nullptr), static_cast<int*>(nullptr),
                                   static_cast<float*>(nullptr), 4, 16, 2),
                 std::invalid_argument);
    EXPECT_THROW(launch_moe_router(static_cast<float*>(nullptr), static_cast<int*>(nullptr),
                                   static_cast<float*>(nullptr), 4, 0, 2),
                 std::invalid_argument);
}

TEST_F(AttentionTest, MoeRouterTopKOutOfRangeThrows) {
    // top_k must satisfy 1 <= top_k <= num_experts.
    EXPECT_THROW(launch_moe_router(static_cast<float*>(nullptr), static_cast<int*>(nullptr),
                                   static_cast<float*>(nullptr), 4, 4, 0),
                 std::invalid_argument);
    EXPECT_THROW(launch_moe_router(static_cast<float*>(nullptr), static_cast<int*>(nullptr),
                                   static_cast<float*>(nullptr), 4, 4, 5),
                 std::invalid_argument);
}

TEST_F(AttentionTest, MoeRouterEmptyBatchReturns) {
    // batch_size == 0 is a no-op, not an error.
    launch_moe_router(static_cast<float*>(nullptr), static_cast<int*>(nullptr),
                      static_cast<float*>(nullptr), 0, 4, 2);
    TC_CUDA_CHECK(cudaGetLastError());
}
