/**
 * @file test_gemm.cpp
 * @brief Tests for GEMM kernels
 */

#include <cuda_fp16.h>

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "tensorcraft/core/cuda_check.hpp"
#include "tensorcraft/kernels/gemm.hpp"

using namespace tensorcraft;
using namespace tensorcraft::kernels;

class GemmTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        gen = std::mt19937(rd());
        dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    }

    std::vector<float> random_matrix(int rows, int cols) {
        std::vector<float> m(rows * cols);
        for (auto& x : m)
            x = dist(gen);
        return m;
    }

    std::vector<float> reference_gemm(const std::vector<float>& A, const std::vector<float>& B,
                                      int M, int N, int K, float alpha, float beta,
                                      const std::vector<float>& C = {}) {
        std::vector<float> result(M * N, 0.0f);

        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A[m * K + k] * B[k * N + n];
                }
                float c_val = C.empty() ? 0.0f : C[m * N + n];
                result[m * N + n] = alpha * sum + beta * c_val;
            }
        }

        return result;
    }

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
};

TEST_F(GemmTest, NaiveCorrectness) {
    const int M = 64, N = 64, K = 64;

    auto h_A = random_matrix(M, K);
    auto h_B = random_matrix(K, N);
    std::vector<float> h_C(M * N, 0.0f);
    auto h_ref = reference_gemm(h_A, h_B, M, N, K, 1.0f, 0.0f);

    float *d_A, *d_B, *d_C;
    TC_CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    TC_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));

    launch_gemm(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, GemmVersion::Naive);
    TC_CUDA_CHECK(cudaDeviceSynchronize());

    TC_CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(h_C[i], h_ref[i], 1e-3f);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(GemmTest, TiledCorrectness) {
    const int M = 128, N = 128, K = 128;

    auto h_A = random_matrix(M, K);
    auto h_B = random_matrix(K, N);
    std::vector<float> h_C(M * N, 0.0f);
    auto h_ref = reference_gemm(h_A, h_B, M, N, K, 1.0f, 0.0f);

    float *d_A, *d_B, *d_C;
    TC_CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    TC_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));

    launch_gemm(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, GemmVersion::Tiled);
    TC_CUDA_CHECK(cudaDeviceSynchronize());

    TC_CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(h_C[i], h_ref[i], 1e-3f);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(GemmTest, DoubleBufferCorrectness) {
    const int M = 128, N = 128, K = 128;

    auto h_A = random_matrix(M, K);
    auto h_B = random_matrix(K, N);
    std::vector<float> h_C(M * N, 0.0f);
    auto h_ref = reference_gemm(h_A, h_B, M, N, K, 1.0f, 0.0f);

    float *d_A, *d_B, *d_C;
    TC_CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    TC_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));

    launch_gemm(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, GemmVersion::DoubleBuffer);
    TC_CUDA_CHECK(cudaDeviceSynchronize());

    TC_CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(h_C[i], h_ref[i], 1e-3f);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(GemmTest, VersionEquivalence) {
    // Property 5: All GEMM versions should produce equivalent results
    const int M = 96, N = 96, K = 96;

    auto h_A = random_matrix(M, K);
    auto h_B = random_matrix(K, N);

    float *d_A, *d_B, *d_C;
    TC_CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    TC_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<std::vector<float>> results;
    std::vector<GemmVersion> versions = {GemmVersion::Naive, GemmVersion::Tiled,
                                         GemmVersion::DoubleBuffer};

    for (auto ver : versions) {
        TC_CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
        launch_gemm(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, ver);
        TC_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_C(M * N);
        TC_CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        results.push_back(h_C);
    }

    // Compare all versions against naive
    for (size_t v = 1; v < results.size(); ++v) {
        for (int i = 0; i < M * N; ++i) {
            EXPECT_NEAR(results[v][i], results[0][i], 1e-3f);
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(GemmTest, NonSquareMatrices) {
    const int M = 64, N = 128, K = 96;

    auto h_A = random_matrix(M, K);
    auto h_B = random_matrix(K, N);
    std::vector<float> h_C(M * N, 0.0f);
    auto h_ref = reference_gemm(h_A, h_B, M, N, K, 1.0f, 0.0f);

    float *d_A, *d_B, *d_C;
    TC_CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    TC_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));

    launch_gemm(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, GemmVersion::Tiled);
    TC_CUDA_CHECK(cudaDeviceSynchronize());

    TC_CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(h_C[i], h_ref[i], 1e-3f);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

#ifdef TC_HAS_WMMA
TEST_F(GemmTest, WmmaCorrectness) {
    const int M = 64, N = 64, K = 64;

    auto h_A = random_matrix(M, K);
    auto h_B = random_matrix(K, N);
    auto h_ref = reference_gemm(h_A, h_B, M, N, K, 1.0f, 0.0f);

    std::vector<half> h_A_half(M * K), h_B_half(K * N);
    for (int i = 0; i < M * K; ++i)
        h_A_half[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; ++i)
        h_B_half[i] = __float2half(h_B[i]);

    half *d_A, *d_B;
    float* d_C;
    TC_CUDA_CHECK(cudaMalloc(&d_A, h_A_half.size() * sizeof(half)));
    TC_CUDA_CHECK(cudaMalloc(&d_B, h_B_half.size() * sizeof(half)));
    TC_CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    TC_CUDA_CHECK(
        cudaMemcpy(d_A, h_A_half.data(), h_A_half.size() * sizeof(half), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(
        cudaMemcpy(d_B, h_B_half.data(), h_B_half.size() * sizeof(half), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));

    launch_gemm_wmma(d_A, d_B, d_C, M, N, K);
    TC_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_C(M * N);
    TC_CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Inputs are rounded to FP16, so allow a small absolute error in addition
    // to the relative term.
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(h_C[i], h_ref[i], 2e-2f);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(GemmTest, WmmaRejectsNonMultipleDimensions) {
    half *d_A = nullptr, *d_B = nullptr;
    float* d_C = nullptr;
    // No allocation should be touched: the host-side input contract must fail
    // before kernel launch.
    EXPECT_THROW(launch_gemm_wmma(d_A, d_B, d_C, 33, 16, 16), std::invalid_argument);
    EXPECT_THROW(launch_gemm_wmma(d_A, d_B, d_C, 16, 17, 16), std::invalid_argument);
    EXPECT_THROW(launch_gemm_wmma(d_A, d_B, d_C, 16, 16, 15), std::invalid_argument);
}
#endif  // TC_HAS_WMMA

TEST_F(GemmTest, TransposeRoundTrip) {
    // Property 11: transpose(transpose(A)) == A
    const int rows = 64, cols = 128;

    auto h_A = random_matrix(rows, cols);
    std::vector<float> h_AT(cols * rows);
    std::vector<float> h_ATT(rows * cols);

    float *d_A, *d_AT, *d_ATT;
    TC_CUDA_CHECK(cudaMalloc(&d_A, rows * cols * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_AT, cols * rows * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_ATT, rows * cols * sizeof(float)));

    TC_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    transpose(d_A, d_AT, rows, cols);
    transpose(d_AT, d_ATT, cols, rows);
    TC_CUDA_CHECK(cudaDeviceSynchronize());

    TC_CUDA_CHECK(
        cudaMemcpy(h_ATT.data(), d_ATT, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < rows * cols; ++i) {
        EXPECT_NEAR(h_ATT[i], h_A[i], 1e-5f);
    }

    cudaFree(d_A);
    cudaFree(d_AT);
    cudaFree(d_ATT);
}
