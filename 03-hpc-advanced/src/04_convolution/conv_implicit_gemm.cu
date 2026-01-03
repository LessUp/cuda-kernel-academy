#include "conv_implicit_gemm.cuh"
#include "../common/cuda_check.cuh"

namespace hpc::convolution {

template <typename T>
__global__ void conv2d_implicit_gemm_kernel(const T* __restrict__ input,
                                             const T* __restrict__ weight,
                                             T* __restrict__ output,
                                             int batch, int in_c, int out_c,
                                             int in_h, int in_w,
                                             int out_h, int out_w,
                                             int k_h, int k_w,
                                             int stride_h, int stride_w,
                                             int pad_h, int pad_w) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = batch * out_c * out_h * out_w;
    
    if (out_idx >= total_out) return;
    
    int ow = out_idx % out_w;
    int oh = (out_idx / out_w) % out_h;
    int oc = (out_idx / (out_w * out_h)) % out_c;
    int b = out_idx / (out_w * out_h * out_c);
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < in_c; ++ic) {
        for (int kh = 0; kh < k_h; ++kh) {
            for (int kw = 0; kw < k_w; ++kw) {
                int ih = oh * stride_h - pad_h + kh;
                int iw = ow * stride_w - pad_w + kw;
                
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int in_idx = b * (in_c * in_h * in_w) + ic * (in_h * in_w) + ih * in_w + iw;
                    int w_idx = oc * (in_c * k_h * k_w) + ic * (k_h * k_w) + kh * k_w + kw;
                    sum += static_cast<float>(input[in_idx]) * static_cast<float>(weight[w_idx]);
                }
            }
        }
    }
    
    output[out_idx] = static_cast<T>(sum);
}

template <>
void conv2d_implicit_gemm<float>(const float* input, const float* weight, float* output,
                                  const ConvParams& p, cudaStream_t stream) {
    int out_h = (p.in_height + 2 * p.pad_h - p.dilation_h * (p.kernel_h - 1) - 1) / p.stride_h + 1;
    int out_w = (p.in_width + 2 * p.pad_w - p.dilation_w * (p.kernel_w - 1) - 1) / p.stride_w + 1;
    int total = p.batch * p.out_channels * out_h * out_w;
    
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    conv2d_implicit_gemm_kernel<float><<<grid_size, block_size, 0, stream>>>(
        input, weight, output,
        p.batch, p.in_channels, p.out_channels,
        p.in_height, p.in_width, out_h, out_w,
        p.kernel_h, p.kernel_w,
        p.stride_h, p.stride_w,
        p.pad_h, p.pad_w);
    CUDA_CHECK_LAST();
}

} // namespace hpc::convolution
