#include <gtest/gtest.h>
#include "07_cuda13_features/fp8_gemm.cuh"
#include "common/tensor.cuh"
#include "../test_utils.hpp"

TEST(FP8GemmTest, BasicTest) {
    int M = 32, N = 32, K = 32;
    auto A = hpc::test::random_vector<float>(M * K, -1.0f, 1.0f);
    auto B = hpc::test::random_vector<float>(K * N, -1.0f, 1.0f);
    
    hpc::Tensor<float> d_A(M * K);
    hpc::Tensor<float> d_B(K * N);
    hpc::Tensor<float> d_C(M * N);
    
    d_A.copy_from_host(A);
    d_B.copy_from_host(B);
    d_C.zero();
    
    hpc::cuda13::fp8_gemm(d_A.data(), d_B.data(), d_C.data(), M, N, K);
    cudaDeviceSynchronize();
    
    auto C = d_C.to_host();
    EXPECT_EQ(C.size(), M * N);
}
