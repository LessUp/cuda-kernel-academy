#include <gtest/gtest.h>
#include "04_convolution/conv_implicit_gemm.cuh"
#include "common/tensor.cuh"
#include "../test_utils.hpp"

TEST(ConvolutionTest, BasicConv2d) {
    int batch = 1, in_c = 3, out_c = 16;
    int in_h = 32, in_w = 32;
    int k_h = 3, k_w = 3;
    int stride = 1, pad = 1;
    
    int out_h = (in_h + 2 * pad - k_h) / stride + 1;
    int out_w = (in_w + 2 * pad - k_w) / stride + 1;
    
    auto input = hpc::test::random_vector<float>(batch * in_c * in_h * in_w, -1.0f, 1.0f);
    auto weight = hpc::test::random_vector<float>(out_c * in_c * k_h * k_w, -1.0f, 1.0f);
    
    hpc::Tensor<float> d_input(batch * in_c * in_h * in_w);
    hpc::Tensor<float> d_weight(out_c * in_c * k_h * k_w);
    hpc::Tensor<float> d_output(batch * out_c * out_h * out_w);
    
    d_input.copy_from_host(input);
    d_weight.copy_from_host(weight);
    
    hpc::convolution::ConvParams params{
        batch, in_c, out_c, in_h, in_w,
        k_h, k_w, stride, stride, pad, pad, 1, 1
    };
    
    hpc::convolution::conv2d_implicit_gemm<float>(
        d_input.data(), d_weight.data(), d_output.data(), params);
    cudaDeviceSynchronize();
    
    auto output = d_output.to_host();
    EXPECT_EQ(output.size(), batch * out_c * out_h * out_w);
}
