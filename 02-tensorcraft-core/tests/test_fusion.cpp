/**
 * @file test_fusion.cpp
 * @brief Basic correctness tests for fused GEMM epilogues
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "tensorcraft/core/cuda_check.hpp"
#include "tensorcraft/kernels/fusion.hpp"

using namespace tensorcraft::kernels;

namespace {

enum class EpilogueKind { Identity, Bias, BiasReLU, BiasGeLU };

// CPU reference for launch_gemm_fused with a given epilogue.
std::vector<float> reference_fused_gemm(const std::vector<float>& A, const std::vector<float>& B,
                                        const std::vector<float>& bias, int M, int N, int K,
                                        EpilogueKind kind) {
    std::vector<float> C(M * N, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            float b = bias.empty() ? 0.0f : bias[n];
            float val = sum + b;
            switch (kind) {
                case EpilogueKind::Identity:
                    val = sum;
                    break;
                case EpilogueKind::Bias:
                    val = sum + b;
                    break;
                case EpilogueKind::BiasReLU:
                    val = std::max(0.0f, sum + b);
                    break;
                case EpilogueKind::BiasGeLU: {
                    float x = sum + b;
                    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
                    constexpr float coeff = 0.044715f;
                    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
                    val = 0.5f * x * (1.0f + std::tanh(inner));
                    break;
                }
            }
            C[m * N + n] = val;
        }
    }
    return C;
}

}  // namespace

class FusionTest : public ::testing::Test {
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
        std::mt19937 gen(2026);
        std::uniform_real_distribution<float> dist(lo, hi);
        std::vector<float> v(n);
        for (auto& x : v)
            x = dist(gen);
        return v;
    }

    void run_and_compare(const std::vector<float>& A, const std::vector<float>& B,
                         const std::vector<float>& bias, int M, int N, int K, EpilogueKind kind,
                         float tol = 1e-3f) {
        float *d_A, *d_B, *d_C;
        TC_CUDA_CHECK(cudaMalloc(&d_A, A.size() * sizeof(float)));
        TC_CUDA_CHECK(cudaMalloc(&d_B, B.size() * sizeof(float)));
        TC_CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        TC_CUDA_CHECK(cudaMemcpy(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
        TC_CUDA_CHECK(cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

        float* d_bias = nullptr;
        if (!bias.empty()) {
            TC_CUDA_CHECK(cudaMalloc(&d_bias, bias.size() * sizeof(float)));
            TC_CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), bias.size() * sizeof(float),
                                     cudaMemcpyHostToDevice));
        }

        switch (kind) {
            case EpilogueKind::Identity:
                launch_gemm_fused<float>(d_A, d_B, d_C, M, N, K, 1.0f, EpilogueIdentity{});
                break;
            case EpilogueKind::Bias:
                launch_gemm_fused<float>(d_A, d_B, d_C, M, N, K, 1.0f, EpilogueBias<float>(d_bias));
                break;
            case EpilogueKind::BiasReLU:
                launch_gemm_fused<float>(d_A, d_B, d_C, M, N, K, 1.0f,
                                         EpilogueBiasReLU<float>(d_bias));
                break;
            case EpilogueKind::BiasGeLU:
                launch_gemm_fused<float>(d_A, d_B, d_C, M, N, K, 1.0f,
                                         EpilogueBiasGeLU<float>(d_bias));
                break;
        }
        TC_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> actual(M * N);
        TC_CUDA_CHECK(
            cudaMemcpy(actual.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        auto expected = reference_fused_gemm(A, B, bias, M, N, K, kind);
        for (int i = 0; i < M * N; ++i) {
            EXPECT_NEAR(actual[i], expected[i], tol) << "element " << i;
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        if (d_bias)
            cudaFree(d_bias);
    }
};

TEST_F(FusionTest, IdentityMatchesCpu) {
    const int M = 32, N = 32, K = 32;
    auto A = random_data(M * K);
    auto B = random_data(K * N);
    run_and_compare(A, B, {}, M, N, K, EpilogueKind::Identity);
}

TEST_F(FusionTest, BiasMatchesCpu) {
    const int M = 33, N = 28, K = 24;  // non-multiple-of-32 to exercise bounds
    auto A = random_data(M * K);
    auto B = random_data(K * N);
    auto bias = random_data(N, -0.5f, 0.5f);
    run_and_compare(A, B, bias, M, N, K, EpilogueKind::Bias);
}

TEST_F(FusionTest, BiasReluMatchesCpu) {
    const int M = 32, N = 40, K = 16;
    auto A = random_data(M * K, -0.5f, 0.5f);
    auto B = random_data(K * N, -0.5f, 0.5f);
    auto bias = random_data(N, -0.3f, 0.3f);
    run_and_compare(A, B, bias, M, N, K, EpilogueKind::BiasReLU);
}

TEST_F(FusionTest, BiasGeluMatchesCpu) {
    const int M = 16, N = 48, K = 32;
    auto A = random_data(M * K, -0.5f, 0.5f);
    auto B = random_data(K * N, -0.5f, 0.5f);
    auto bias = random_data(N, -0.3f, 0.3f);
    run_and_compare(A, B, bias, M, N, K, EpilogueKind::BiasGeLU);
}
