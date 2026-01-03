#pragma once
/**
 * @file conv2d.hpp
 * @brief 2D Convolution kernels with multiple implementations
 */

#include "../core/features.hpp"
#include "../core/cuda_check.hpp"
#include "../core/type_traits.hpp"

namespace tensorcraft {
namespace kernels {

// Naive Conv2D kernel
template<typename T>
__global__ void conv2d_naive_kernel(
    const T* TC_RESTRICT input,
    const T* TC_RESTRICT weight,
    const T* TC_RESTRICT bias,
    T* TC_RESTRICT output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int OH, int OW,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {
    
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z % K;
    const int n = blockIdx.z / K;
    
    if (ow >= OW || oh >= OH) return;
    
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int ih = oh * stride_h - pad_h + r;
                int iw = ow * stride_w - pad_w + s;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float in_val = to_float(input[((n * C + c) * H + ih) * W + iw]);
                    float w_val = to_float(weight[((k * C + c) * R + r) * S + s]);
                    sum += in_val * w_val;
                }
            }
        }
    }
    if (bias) sum += to_float(bias[k]);
    output[((n * K + k) * OH + oh) * OW + ow] = from_float<T>(sum);
}

// Im2Col kernel
template<typename T>
__global__ void im2col_kernel(
    const T* TC_RESTRICT input,
    T* TC_RESTRICT col,
    int N, int C, int H, int W,
    int R, int S, int OH, int OW,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * R * S * OH * OW;
    if (idx >= total) return;
    
    int ow = idx % OW;
    int remaining = idx / OW;
    int oh = remaining % OH;
    remaining /= OH;
    int s = remaining % S;
    remaining /= S;
    int r = remaining % R;
    remaining /= R;
    int c = remaining % C;
    int n = remaining / C;
    
    int ih = oh * stride_h - pad_h + r;
    int iw = ow * stride_w - pad_w + s;
    
    T val = T(0);
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        val = input[((n * C + c) * H + ih) * W + iw];
    }
    int col_idx = (n * (C * R * S) + (c * R * S + r * S + s)) * (OH * OW) + (oh * OW + ow);
    col[col_idx] = val;
}

// Depthwise Conv2D kernel
template<typename T>
__global__ void conv2d_depthwise_kernel(
    const T* TC_RESTRICT input,
    const T* TC_RESTRICT weight,
    const T* TC_RESTRICT bias,
    T* TC_RESTRICT output,
    int N, int C, int H, int W,
    int R, int S, int OH, int OW,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {
    
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % C;
    int n = blockIdx.z / C;
    
    if (ow >= OW || oh >= OH) return;
    
    float sum = 0.0f;
    for (int r = 0; r < R; ++r) {
        for (int s = 0; s < S; ++s) {
            int ih = oh * stride_h - pad_h + r;
            int iw = ow * stride_w - pad_w + s;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                float in_val = to_float(input[((n * C + c) * H + ih) * W + iw]);
                float w_val = to_float(weight[(c * R + r) * S + s]);
                sum += in_val * w_val;
            }
        }
    }
    if (bias) sum += to_float(bias[c]);
    output[((n * C + c) * OH + oh) * OW + ow] = from_float<T>(sum);
}

// Pointwise (1x1) Conv2D kernel
template<typename T>
__global__ void conv2d_pointwise_kernel(
    const T* TC_RESTRICT input,
    const T* TC_RESTRICT weight,
    const T* TC_RESTRICT bias,
    T* TC_RESTRICT output,
    int N, int C, int H, int W, int K) {
    
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int n = blockIdx.z;
    int HW = H * W;
    
    if (spatial_idx >= HW) return;
    
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        float in_val = to_float(input[(n * C + c) * HW + spatial_idx]);
        float w_val = to_float(weight[k * C + c]);
        sum += in_val * w_val;
    }
    if (bias) sum += to_float(bias[k]);
    output[(n * K + k) * HW + spatial_idx] = from_float<T>(sum);
}

// Launcher functions
template<typename T>
void launch_conv2d_naive(
    const T* input, const T* weight, const T* bias, T* output,
    int N, int C, int H, int W, int K, int R, int S,
    int stride_h, int stride_w, int pad_h, int pad_w,
    cudaStream_t stream = nullptr) {
    
    int OH = (H + 2 * pad_h - R) / stride_h + 1;
    int OW = (W + 2 * pad_w - S) / stride_w + 1;
    dim3 block(16, 16);
    dim3 grid((OW + 15) / 16, (OH + 15) / 16, N * K);
    conv2d_naive_kernel<T><<<grid, block, 0, stream>>>(
        input, weight, bias, output, N, C, H, W, K, R, S, OH, OW,
        stride_h, stride_w, pad_h, pad_w);
    TC_CUDA_CHECK_LAST();
}

template<typename T>
void launch_im2col(
    const T* input, T* col,
    int N, int C, int H, int W, int R, int S,
    int stride_h, int stride_w, int pad_h, int pad_w,
    cudaStream_t stream = nullptr) {
    
    int OH = (H + 2 * pad_h - R) / stride_h + 1;
    int OW = (W + 2 * pad_w - S) / stride_w + 1;
    int total = N * C * R * S * OH * OW;
    int grid_size = (total + 255) / 256;
    im2col_kernel<T><<<grid_size, 256, 0, stream>>>(
        input, col, N, C, H, W, R, S, OH, OW, stride_h, stride_w, pad_h, pad_w);
    TC_CUDA_CHECK_LAST();
}

template<typename T>
void launch_conv2d_depthwise(
    const T* input, const T* weight, const T* bias, T* output,
    int N, int C, int H, int W, int R, int S,
    int stride_h, int stride_w, int pad_h, int pad_w,
    cudaStream_t stream = nullptr) {
    
    int OH = (H + 2 * pad_h - R) / stride_h + 1;
    int OW = (W + 2 * pad_w - S) / stride_w + 1;
    dim3 block(16, 16);
    dim3 grid((OW + 15) / 16, (OH + 15) / 16, N * C);
    conv2d_depthwise_kernel<T><<<grid, block, 0, stream>>>(
        input, weight, bias, output, N, C, H, W, R, S, OH, OW,
        stride_h, stride_w, pad_h, pad_w);
    TC_CUDA_CHECK_LAST();
}

template<typename T>
void launch_conv2d_pointwise(
    const T* input, const T* weight, const T* bias, T* output,
    int N, int C, int H, int W, int K,
    cudaStream_t stream = nullptr) {
    
    int HW = H * W;
    dim3 block(256);
    dim3 grid((HW + 255) / 256, K, N);
    conv2d_pointwise_kernel<T><<<grid, block, 0, stream>>>(
        input, weight, bias, output, N, C, H, W, K);
    TC_CUDA_CHECK_LAST();
}

// Convenience functions
template<typename T>
void conv2d(const T* input, const T* weight, const T* bias, T* output,
            size_t N, size_t C, size_t H, size_t W,
            size_t K, size_t R, size_t S,
            int stride = 1, int padding = 0, cudaStream_t stream = nullptr) {
    launch_conv2d_naive(input, weight, bias, output,
        (int)N, (int)C, (int)H, (int)W, (int)K, (int)R, (int)S,
        stride, stride, padding, padding, stream);
}

template<typename T>
void conv2d_depthwise(const T* input, const T* weight, const T* bias, T* output,
                      size_t N, size_t C, size_t H, size_t W, size_t R, size_t S,
                      int stride = 1, int padding = 0, cudaStream_t stream = nullptr) {
    launch_conv2d_depthwise(input, weight, bias, output,
        (int)N, (int)C, (int)H, (int)W, (int)R, (int)S,
        stride, stride, padding, padding, stream);
}

} // namespace kernels
} // namespace tensorcraft
