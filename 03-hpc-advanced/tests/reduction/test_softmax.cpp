#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include "02_reduction/softmax.cuh"
#include "common/tensor.cuh"
#include "../test_utils.hpp"
#include <cmath>
#include <numeric>

// Feature: hpc-ai-optimization-lab, Property 6: Softmax Output Properties
RC_GTEST_PROP(SoftmaxTest, OutputProperties, ()) {
    auto batch = *rc::gen::inRange<int>(1, 32);
    auto seq_len = *rc::gen::inRange<int>(32, 512);
    auto input = *rc::gen::container<std::vector<float>>(batch * seq_len,
        rc::gen::map(rc::gen::arbitrary<float>(), [](float x) {
            return std::clamp(x, -10.0f, 10.0f);
        }));
    
    hpc::Tensor<float> d_input(batch * seq_len);
    hpc::Tensor<float> d_output(batch * seq_len);
    d_input.copy_from_host(input);
    
    hpc::reduction::softmax<float, hpc::reduction::SoftmaxOpt::OnlineSoftmax>(
        d_input.data(), d_output.data(), batch, seq_len);
    cudaDeviceSynchronize();
    
    auto result = d_output.to_host();
    
    for (int b = 0; b < batch; ++b) {
        float sum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            float val = result[b * seq_len + i];
            // Property 1: All values in [0, 1]
            RC_ASSERT(val >= 0.0f && val <= 1.0f);
            sum += val;
        }
        // Property 2: Sum to 1
        RC_ASSERT(hpc::test::almost_equal(sum, 1.0f, 1e-3f, 1e-4f));
    }
}

TEST(SoftmaxTest, BasicTest) {
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    
    hpc::Tensor<float> d_input(4);
    hpc::Tensor<float> d_output(4);
    d_input.copy_from_host(input);
    
    hpc::reduction::softmax<float, hpc::reduction::SoftmaxOpt::OnlineSoftmax>(
        d_input.data(), d_output.data(), 1, 4);
    cudaDeviceSynchronize();
    
    auto result = d_output.to_host();
    float sum = std::accumulate(result.begin(), result.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-4f);
}
