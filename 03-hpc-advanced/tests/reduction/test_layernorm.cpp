#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include "02_reduction/layernorm.cuh"
#include "02_reduction/rmsnorm.cuh"
#include "common/tensor.cuh"
#include "../test_utils.hpp"
#include <cmath>
#include <numeric>

// Feature: hpc-ai-optimization-lab, Property 7: LayerNorm/RMSNorm Output Properties
RC_GTEST_PROP(LayerNormTest, OutputProperties, ()) {
    auto batch = *rc::gen::inRange<int>(1, 16);
    auto hidden = *rc::gen::inRange<int>(64, 512);
    auto input = *rc::gen::container<std::vector<float>>(batch * hidden,
        rc::gen::map(rc::gen::arbitrary<float>(), [](float x) {
            return std::clamp(x, -10.0f, 10.0f);
        }));
    
    // Gamma = 1, Beta = 0 for testing normalized output
    std::vector<float> gamma(hidden, 1.0f);
    std::vector<float> beta(hidden, 0.0f);
    
    hpc::Tensor<float> d_input(batch * hidden);
    hpc::Tensor<float> d_gamma(hidden);
    hpc::Tensor<float> d_beta(hidden);
    hpc::Tensor<float> d_output(batch * hidden);
    
    d_input.copy_from_host(input);
    d_gamma.copy_from_host(gamma);
    d_beta.copy_from_host(beta);
    
    hpc::reduction::layer_norm<float>(
        d_input.data(), d_gamma.data(), d_beta.data(),
        d_output.data(), batch, hidden);
    cudaDeviceSynchronize();
    
    auto result = d_output.to_host();
    
    for (int b = 0; b < batch; ++b) {
        float mean = 0.0f;
        for (int i = 0; i < hidden; ++i) {
            mean += result[b * hidden + i];
        }
        mean /= hidden;
        
        // Mean should be close to 0 (beta)
        RC_ASSERT(hpc::test::almost_equal(mean, 0.0f, 1e-2f, 1e-3f));
    }
}

TEST(LayerNormTest, BasicTest) {
    int batch = 2;
    int hidden = 64;
    auto input = hpc::test::random_vector<float>(batch * hidden);
    std::vector<float> gamma(hidden, 1.0f);
    std::vector<float> beta(hidden, 0.0f);
    
    hpc::Tensor<float> d_input(batch * hidden);
    hpc::Tensor<float> d_gamma(hidden);
    hpc::Tensor<float> d_beta(hidden);
    hpc::Tensor<float> d_output(batch * hidden);
    
    d_input.copy_from_host(input);
    d_gamma.copy_from_host(gamma);
    d_beta.copy_from_host(beta);
    
    hpc::reduction::layer_norm<float>(
        d_input.data(), d_gamma.data(), d_beta.data(),
        d_output.data(), batch, hidden);
    cudaDeviceSynchronize();
    
    auto result = d_output.to_host();
    EXPECT_EQ(result.size(), batch * hidden);
}
