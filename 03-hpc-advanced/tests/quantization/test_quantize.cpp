#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include "06_quantization/int8_quant.cuh"
#include "06_quantization/dequant.cuh"
#include "common/tensor.cuh"
#include "../test_utils.hpp"

// Feature: hpc-ai-optimization-lab, Property 16: Quantization Round Trip
RC_GTEST_PROP(QuantizationTest, RoundTrip, ()) {
    auto rows = *rc::gen::inRange<int>(1, 64);
    auto cols = *rc::gen::inRange<int>(64, 256);
    auto input = *rc::gen::container<std::vector<float>>(rows * cols,
        rc::gen::map(rc::gen::arbitrary<float>(), [](float x) {
            return std::clamp(x, -1.0f, 1.0f);
        }));
    
    hpc::Tensor<float> d_input(rows * cols);
    hpc::Tensor<int8_t> d_quantized(rows * cols);
    hpc::Tensor<float> d_scale(rows);
    hpc::Tensor<float> d_output(rows * cols);
    
    d_input.copy_from_host(input);
    
    // Quantize
    hpc::quantization::quantize_int8(
        d_input.data(), d_quantized.data(), d_scale.data(), rows, cols);
    
    // Dequantize
    hpc::quantization::dequantize_int8(
        d_quantized.data(), d_scale.data(), d_output.data(), rows, cols);
    cudaDeviceSynchronize();
    
    auto result = d_output.to_host();
    
    // Check that round-trip error is bounded
    for (size_t i = 0; i < input.size(); ++i) {
        float error = std::abs(result[i] - input[i]);
        // INT8 quantization error should be bounded by scale/127
        RC_ASSERT(error < 0.1f);  // Reasonable error bound
    }
}

TEST(QuantizationTest, BasicRoundTrip) {
    int rows = 4, cols = 64;
    auto input = hpc::test::random_vector<float>(rows * cols, -1.0f, 1.0f);
    
    hpc::Tensor<float> d_input(rows * cols);
    hpc::Tensor<int8_t> d_quantized(rows * cols);
    hpc::Tensor<float> d_scale(rows);
    hpc::Tensor<float> d_output(rows * cols);
    
    d_input.copy_from_host(input);
    
    hpc::quantization::quantize_int8(
        d_input.data(), d_quantized.data(), d_scale.data(), rows, cols);
    hpc::quantization::dequantize_int8(
        d_quantized.data(), d_scale.data(), d_output.data(), rows, cols);
    cudaDeviceSynchronize();
    
    auto result = d_output.to_host();
    
    // Check reasonable reconstruction
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(result[i], input[i], 0.1f);
    }
}
