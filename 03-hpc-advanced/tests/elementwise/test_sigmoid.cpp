#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include "01_elementwise/sigmoid.cuh"
#include "common/tensor.cuh"
#include "../test_utils.hpp"
#include <cmath>

RC_GTEST_PROP(SigmoidTest, Correctness, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 1024 * 64);
    auto input = *rc::gen::container<std::vector<float>>(size, 
        rc::gen::map(rc::gen::arbitrary<float>(), [](float x) {
            return std::clamp(x, -10.0f, 10.0f);
        }));
    
    // CPU reference
    std::vector<float> expected(size);
    for (size_t i = 0; i < size; ++i) {
        expected[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
    
    // GPU implementation
    hpc::Tensor<float> d_input(size);
    hpc::Tensor<float> d_output(size);
    d_input.copy_from_host(input);
    
    hpc::elementwise::sigmoid<float, hpc::elementwise::OptLevel::GridStride>(
        d_input.data(), d_output.data(), size);
    cudaDeviceSynchronize();
    
    auto result = d_output.to_host();
    
    for (size_t i = 0; i < size; ++i) {
        RC_ASSERT(hpc::test::almost_equal(result[i], expected[i], 1e-4f, 1e-5f));
    }
}

TEST(SigmoidTest, BasicTest) {
    std::vector<float> input = {0.0f, 1.0f, -1.0f};
    
    hpc::Tensor<float> d_input(input.size());
    hpc::Tensor<float> d_output(input.size());
    d_input.copy_from_host(input);
    
    hpc::elementwise::sigmoid<float, hpc::elementwise::OptLevel::GridStride>(
        d_input.data(), d_output.data(), input.size());
    cudaDeviceSynchronize();
    
    auto result = d_output.to_host();
    EXPECT_NEAR(result[0], 0.5f, 1e-5f);
}
