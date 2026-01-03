#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace hpc::convolution {

struct ConvParams {
    int batch;
    int in_channels;
    int out_channels;
    int in_height;
    int in_width;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int dilation_h;
    int dilation_w;
};

template <typename T>
void conv2d_implicit_gemm(const T* input, const T* weight, T* output,
                          const ConvParams& params, cudaStream_t stream = nullptr);

} // namespace hpc::convolution
