#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include "07_cuda13_features/tma.cuh"
#include "common/tensor.cuh"
#include "../test_utils.hpp"

// Feature: hpc-ai-optimization-lab, Property 13: TMA Data Integrity
RC_GTEST_PROP(TMATest, DataIntegrity, ()) {
    auto rows = *rc::gen::inRange<int>(1, 128);
    auto cols = *rc::gen::inRange<int>(1, 128);
    auto input = *rc::gen::container<std::vector<float>>(rows * cols, rc::gen::arbitrary<float>());
    
    hpc::Tensor<float> d_src(rows * cols);
    hpc::Tensor<float> d_dst(rows * cols);
    d_src.copy_from_host(input);
    
    hpc::cuda13::tma_copy_2d<float>(d_src.data(), d_dst.data(), rows, cols);
    cudaDeviceSynchronize();
    
    auto result = d_dst.to_host();
    
    for (size_t i = 0; i < input.size(); ++i) {
        RC_ASSERT(input[i] == result[i]);
    }
}

TEST(TMATest, BasicCopy) {
    int rows = 64, cols = 64;
    auto input = hpc::test::random_vector<float>(rows * cols);
    
    hpc::Tensor<float> d_src(rows * cols);
    hpc::Tensor<float> d_dst(rows * cols);
    d_src.copy_from_host(input);
    
    hpc::cuda13::tma_copy_2d<float>(d_src.data(), d_dst.data(), rows, cols);
    cudaDeviceSynchronize();
    
    auto result = d_dst.to_host();
    EXPECT_TRUE(hpc::test::vectors_almost_equal(result, input));
}
