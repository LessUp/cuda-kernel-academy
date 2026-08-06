#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>

#include "../test_utils.hpp"
#include "02_reduction/layernorm.cuh"
#include "02_reduction/rmsnorm.cuh"
#include "common/tensor.cuh"

// Feature: hpc-ai-optimization-lab, Property 7: LayerNorm/RMSNorm Output Properties
RC_GTEST_PROP(LayerNormTest, OutputProperties, ()) {
    auto batch = *rc::gen::inRange<int>(1, 16);
    auto hidden = *rc::gen::inRange<int>(64, 512);
    auto input = *hpc::test::gen::sized_float_vector(static_cast<size_t>(batch * hidden));
    for (auto& value : input) {
        value = std::clamp(value, -10.0f, 10.0f);
    }

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

    hpc::reduction::layer_norm<float>(d_input.data(), d_gamma.data(), d_beta.data(),
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

RC_GTEST_PROP(RMSNormTest, MatchesCpuReference, ()) {
    auto batch = *rc::gen::inRange<int>(1, 8);
    auto hidden = *rc::gen::inRange<int>(64, 512);
    auto input = *hpc::test::gen::sized_float_vector(static_cast<size_t>(batch * hidden));
    for (auto& value : input) {
        value = std::clamp(value, -10.0f, 10.0f);
    }

    constexpr float eps = 1e-5f;
    std::vector<float> gamma(hidden, 1.0f);
    hpc::Tensor<float> d_input(batch * hidden);
    hpc::Tensor<float> d_gamma(hidden);
    hpc::Tensor<float> d_output(batch * hidden);

    d_input.copy_from_host(input);
    d_gamma.copy_from_host(gamma);
    hpc::reduction::rms_norm<float>(d_input.data(), d_gamma.data(), d_output.data(), batch, hidden,
                                    eps);
    cudaDeviceSynchronize();

    auto result = d_output.to_host();
    for (int row = 0; row < batch; ++row) {
        double sum_sq = 0.0;
        for (int col = 0; col < hidden; ++col) {
            double value = input[row * hidden + col];
            sum_sq += value * value;
        }
        double inv_rms = 1.0 / std::sqrt(sum_sq / hidden + eps);
        for (int col = 0; col < hidden; ++col) {
            float expected = static_cast<float>(input[row * hidden + col] * inv_rms);
            RC_ASSERT(hpc::test::almost_equal(result[row * hidden + col], expected, 2e-3f, 2e-4f));
        }
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

    hpc::reduction::layer_norm<float>(d_input.data(), d_gamma.data(), d_beta.data(),
                                      d_output.data(), batch, hidden);
    cudaDeviceSynchronize();

    auto result = d_output.to_host();
    EXPECT_EQ(result.size(), batch * hidden);
}
