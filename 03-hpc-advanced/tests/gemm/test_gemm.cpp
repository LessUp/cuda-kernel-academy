#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include "03_gemm/gemm.cuh"
#include "common/tensor.cuh"
#include "../test_utils.hpp"

// CPU reference GEMM
void cpu_gemm(const float* A, const float* B, float* C,
              int M, int N, int K, float alpha, float beta) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

// Feature: hpc-ai-optimization-lab, Property 8: GEMM Correctness
RC_GTEST_PROP(GemmTest, Correctness, ()) {
    auto M = *rc::gen::inRange<int>(1, 64);
    auto N = *rc::gen::inRange<int>(1, 64);
    auto K = *rc::gen::inRange<int>(1, 64);
    
    auto A = *rc::gen::container<std::vector<float>>(M * K,
        rc::gen::map(rc::gen::arbitrary<float>(), [](float x) {
            return std::clamp(x, -1.0f, 1.0f);
        }));
    auto B = *rc::gen::container<std::vector<float>>(K * N,
        rc::gen::map(rc::gen::arbitrary<float>(), [](float x) {
            return std::clamp(x, -1.0f, 1.0f);
        }));
    
    std::vector<float> C_cpu(M * N, 0.0f);
    std::vector<float> C_gpu(M * N, 0.0f);
    
    // CPU reference
    cpu_gemm(A.data(), B.data(), C_cpu.data(), M, N, K, 1.0f, 0.0f);
    
    // GPU implementation
    hpc::Tensor<float> d_A(M * K);
    hpc::Tensor<float> d_B(K * N);
    hpc::Tensor<float> d_C(M * N);
    d_A.copy_from_host(A);
    d_B.copy_from_host(B);
    d_C.copy_from_host(C_gpu);
    
    hpc::gemm::gemm<float, hpc::gemm::GemmOpt::SharedMemTiling>(
        d_A.data(), d_B.data(), d_C.data(), M, N, K);
    cudaDeviceSynchronize();
    
    C_gpu = d_C.to_host();
    
    for (int i = 0; i < M * N; ++i) {
        RC_ASSERT(hpc::test::almost_equal(C_gpu[i], C_cpu[i], 1e-3f, 1e-4f));
    }
}

TEST(GemmTest, BasicTest) {
    int M = 32, N = 32, K = 32;
    auto A = hpc::test::random_vector<float>(M * K, -1.0f, 1.0f);
    auto B = hpc::test::random_vector<float>(K * N, -1.0f, 1.0f);
    std::vector<float> C_cpu(M * N, 0.0f);
    
    cpu_gemm(A.data(), B.data(), C_cpu.data(), M, N, K, 1.0f, 0.0f);
    
    hpc::Tensor<float> d_A(M * K);
    hpc::Tensor<float> d_B(K * N);
    hpc::Tensor<float> d_C(M * N);
    d_A.copy_from_host(A);
    d_B.copy_from_host(B);
    d_C.zero();
    
    hpc::gemm::gemm<float, hpc::gemm::GemmOpt::Naive>(
        d_A.data(), d_B.data(), d_C.data(), M, N, K);
    cudaDeviceSynchronize();
    
    auto C_gpu = d_C.to_host();
    EXPECT_TRUE(hpc::test::vectors_almost_equal(C_gpu, C_cpu, 1e-3f, 1e-4f));
}
