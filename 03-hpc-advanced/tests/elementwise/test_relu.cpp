#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include "01_elementwise/relu.cuh"
#include "common/tensor.cuh"
#include "../test_utils.hpp"
#include <algorithm>

// Feature: hpc-ai-optimization-lab, Property 3: Elementwise Operation Correctness
RC_GTEST_PROP(ReluTest, Correctness, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 1024 * 64);
    auto input = *rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>());
    
    // CPU reference
    std::vector<float> expected(size);
    for (size_t i = 0; i < size; ++i) {
        expected[i] = std::max(0.0f, input[i]);
    }
    
    // GPU implementation
    hpc::Tensor<float> d_input(size);
    hpc::Tensor<float> d_output(size);
    d_input.copy_from_host(input);
    
    hpc::elementwise::relu<float, hpc::elementwise::OptLevel::GridStride>(
        d_input.data(), d_output.data(), size);
    cudaDeviceSynchronize();
    
    auto result = d_output.to_host();
    
    for (size_t i = 0; i < size; ++i) {
        RC_ASSERT(hpc::test::almost_equal(result[i], expected[i]));
    }
}

TEST(ReluTest, BasicTest) {
    std::vector<float> input = {-1.0f, 0.0f, 1.0f, -0.5f, 0.5f};
    std::vector<float> expected = {0.0f, 0.0f, 1.0f, 0.0f, 0.5f};
    
    hpc::Tensor<float> d_input(input.size());
    hpc::Tensor<float> d_output(input.size());
    d_input.copy_from_host(input);
    
    hpc::elementwise::relu<float, hpc::elementwise::OptLevel::Naive>(
        d_input.data(), d_output.data(), input.size());
    cudaDeviceSynchronize();
    
    auto result = d_output.to_host();
    EXPECT_TRUE(hpc::test::vectors_almost_equal(result, expected));
}
