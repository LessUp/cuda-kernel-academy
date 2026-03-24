#include <gtest/gtest.h>

#include "common.h"
#include "kernels.cuh"

using namespace mini_inference;

class FusionTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices found.";
        }
        CUDA_CHECK(cudaSetDevice(0));
    }
};

// ============================================================================
// Property 3: Kernel Fusion Correctness
// Fused kernel output matches separate operations
// ============================================================================

TEST_F(FusionTest, FusedGemmBiasRelu) {
    // Feature: mini-inference-engine, Property 3: Kernel Fusion Correctness
    const int M = 128, N = 256, K = 64;

    std::vector<float> h_A(M * K), h_B(K * N), h_bias(N);
    std::vector<float> h_C_fused(M * N), h_C_ref(M * N);

    random_init(h_A.data(), M * K);
    random_init(h_B.data(), K * N);
    random_init(h_bias.data(), N, -0.5f, 0.5f);

    // CPU reference: MatMul + Bias + ReLU
    cpu_matmul_bias_relu(h_A.data(), h_B.data(), h_C_ref.data(), h_bias.data(), M, N, K, true,
                         true);

    // GPU fused kernel
    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));
    DeviceMemory d_bias(N * sizeof(float));

    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));
    d_bias.copy_from_host(h_bias.data(), N * sizeof(float));

    launch_fused_gemm(d_A.get(), d_B.get(), d_C.get(), d_bias.get(), M, N, K, true, true);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_C.copy_to_host(h_C_fused.data(), M * N * sizeof(float));

    float max_error = compare_matrices(h_C_fused.data(), h_C_ref.data(), M * N);
    EXPECT_LT(max_error, 1e-4f) << "Fused kernel error: " << max_error;
}

TEST_F(FusionTest, FusedGemmBiasOnly) {
    const int M = 64, N = 128, K = 32;

    std::vector<float> h_A(M * K), h_B(K * N), h_bias(N);
    std::vector<float> h_C_fused(M * N), h_C_ref(M * N);

    random_init(h_A.data(), M * K);
    random_init(h_B.data(), K * N);
    random_init(h_bias.data(), N);

    // CPU reference: MatMul + Bias (no ReLU)
    cpu_matmul_bias_relu(h_A.data(), h_B.data(), h_C_ref.data(), h_bias.data(), M, N, K, true,
                         false);

    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));
    DeviceMemory d_bias(N * sizeof(float));

    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));
    d_bias.copy_from_host(h_bias.data(), N * sizeof(float));

    launch_fused_gemm(d_A.get(), d_B.get(), d_C.get(), d_bias.get(), M, N, K, true, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_C.copy_to_host(h_C_fused.data(), M * N * sizeof(float));

    float max_error = compare_matrices(h_C_fused.data(), h_C_ref.data(), M * N);
    EXPECT_LT(max_error, 1e-4f) << "Fused (bias only) error: " << max_error;
}

TEST_F(FusionTest, FusedGemmReluOnly) {
    const int M = 64, N = 128, K = 32;

    std::vector<float> h_A(M * K), h_B(K * N);
    std::vector<float> h_C_fused(M * N), h_C_ref(M * N);

    random_init(h_A.data(), M * K);
    random_init(h_B.data(), K * N);

    // CPU reference: MatMul + ReLU (no bias)
    cpu_matmul_bias_relu(h_A.data(), h_B.data(), h_C_ref.data(), nullptr, M, N, K, false, true);

    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));

    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));

    launch_fused_gemm(d_A.get(), d_B.get(), d_C.get(), nullptr, M, N, K, false, true);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_C.copy_to_host(h_C_fused.data(), M * N * sizeof(float));

    float max_error = compare_matrices(h_C_fused.data(), h_C_ref.data(), M * N);
    EXPECT_LT(max_error, 1e-4f) << "Fused (ReLU only) error: " << max_error;
}

TEST_F(FusionTest, FusedGemmNoFusion) {
    const int M = 64, N = 128, K = 32;

    std::vector<float> h_A(M * K), h_B(K * N);
    std::vector<float> h_C_fused(M * N), h_C_ref(M * N);

    random_init(h_A.data(), M * K);
    random_init(h_B.data(), K * N);

    // CPU reference: MatMul only
    cpu_matmul(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));

    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));

    launch_fused_gemm(d_A.get(), d_B.get(), d_C.get(), nullptr, M, N, K, false, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_C.copy_to_host(h_C_fused.data(), M * N * sizeof(float));

    float max_error = compare_matrices(h_C_fused.data(), h_C_ref.data(), M * N);
    EXPECT_LT(max_error, 1e-4f) << "Fused (no fusion) error: " << max_error;
}

// ============================================================================
// Fusion vs Separate Kernels Equivalence
// ============================================================================

TEST_F(FusionTest, FusedVsSeparateKernels) {
    const int M = 256, N = 256, K = 128;

    std::vector<float> h_A(M * K), h_B(K * N), h_bias(N);
    std::vector<float> h_C_fused(M * N), h_C_separate(M * N);

    random_init(h_A.data(), M * K);
    random_init(h_B.data(), K * N);
    random_init(h_bias.data(), N);

    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));
    DeviceMemory d_bias(N * sizeof(float));

    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));
    d_bias.copy_from_host(h_bias.data(), N * sizeof(float));

    // Fused kernel
    launch_fused_gemm(d_A.get(), d_B.get(), d_C.get(), d_bias.get(), M, N, K, true, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_C.copy_to_host(h_C_fused.data(), M * N * sizeof(float));

    // Separate operations (using CPU for bias + ReLU)
    d_C.zero();
    launch_optimized_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_C.copy_to_host(h_C_separate.data(), M * N * sizeof(float));

    // Apply bias and ReLU on CPU
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float val = h_C_separate[i * N + j] + h_bias[j];
            h_C_separate[i * N + j] = std::max(0.0f, val);
        }
    }

    float max_error = compare_matrices(h_C_fused.data(), h_C_separate.data(), M * N);
    EXPECT_LT(max_error, 1e-3f) << "Fused vs Separate error: " << max_error;
}
