#include <gtest/gtest.h>

#include "../test_utils.hpp"
#include "04_convolution/conv_implicit_gemm.cuh"
#include "common/tensor.cuh"

namespace {

int output_dim(int input, int kernel, int pad, int stride, int dilation) {
    return (input + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

std::vector<float> reference_conv2d(const std::vector<float>& input,
                                    const std::vector<float>& weight,
                                    const hpc::convolution::ConvParams& params) {
    int out_h = output_dim(params.in_height, params.kernel_h, params.pad_h, params.stride_h,
                           params.dilation_h);
    int out_w = output_dim(params.in_width, params.kernel_w, params.pad_w, params.stride_w,
                           params.dilation_w);
    std::vector<float> output(params.batch * params.out_channels * out_h * out_w, 0.0f);

    for (int b = 0; b < params.batch; ++b) {
        for (int oc = 0; oc < params.out_channels; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < params.in_channels; ++ic) {
                        for (int kh = 0; kh < params.kernel_h; ++kh) {
                            for (int kw = 0; kw < params.kernel_w; ++kw) {
                                int ih =
                                    oh * params.stride_h - params.pad_h + kh * params.dilation_h;
                                int iw =
                                    ow * params.stride_w - params.pad_w + kw * params.dilation_w;
                                if (ih >= 0 && ih < params.in_height && iw >= 0 &&
                                    iw < params.in_width) {
                                    int in_idx = b * (params.in_channels * params.in_height *
                                                      params.in_width) +
                                                 ic * (params.in_height * params.in_width) +
                                                 ih * params.in_width + iw;
                                    int w_idx = oc * (params.in_channels * params.kernel_h *
                                                      params.kernel_w) +
                                                ic * (params.kernel_h * params.kernel_w) +
                                                kh * params.kernel_w + kw;
                                    sum += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                    int out_idx = b * (params.out_channels * out_h * out_w) + oc * (out_h * out_w) +
                                  oh * out_w + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }

    return output;
}

void run_conv_test(int batch, int in_c, int out_c, int in_h, int in_w, int k_h, int k_w, int stride,
                   int pad, int dilation) {
    int out_h = output_dim(in_h, k_h, pad, stride, dilation);
    int out_w = output_dim(in_w, k_w, pad, stride, dilation);

    auto input = hpc::test::random_vector<float>(batch * in_c * in_h * in_w, -1.0f, 1.0f);
    auto weight = hpc::test::random_vector<float>(out_c * in_c * k_h * k_w, -1.0f, 1.0f);

    hpc::Tensor<float> d_input(batch * in_c * in_h * in_w);
    hpc::Tensor<float> d_weight(out_c * in_c * k_h * k_w);
    hpc::Tensor<float> d_output(batch * out_c * out_h * out_w);

    d_input.copy_from_host(input);
    d_weight.copy_from_host(weight);

    hpc::convolution::ConvParams params{batch,  in_c,   out_c, in_h, in_w,     k_h,     k_w,
                                        stride, stride, pad,   pad,  dilation, dilation};

    hpc::convolution::conv2d_implicit_gemm<float>(d_input.data(), d_weight.data(), d_output.data(),
                                                  params);
    cudaDeviceSynchronize();

    auto output = d_output.to_host();
    auto reference = reference_conv2d(input, weight, params);
    EXPECT_TRUE(hpc::test::vectors_almost_equal(output, reference, 1e-4f, 1e-5f));
}

}  // namespace

TEST(ConvolutionTest, BasicConv2d) {
    run_conv_test(1, 3, 16, 32, 32, 3, 3, 1, 1, 1);
}

TEST(ConvolutionTest, DilationConv2d) {
    run_conv_test(1, 2, 4, 10, 10, 3, 3, 1, 2, 2);
}
