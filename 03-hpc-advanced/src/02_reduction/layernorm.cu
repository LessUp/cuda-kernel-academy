#include "layernorm.cuh"
#include "../common/cuda_check.cuh"

namespace hpc::reduction {

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__global__ void layer_norm_kernel(const T* __restrict__ input,
                                   const T* __restrict__ gamma,
                                   const T* __restrict__ beta,
                                   T* __restrict__ output,
                                   int hidden_size, float eps) {
    int row = blockIdx.x;
    const T* row_input = input + row * hidden_size;
    T* row_output = output + row * hidden_size;

    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sum += static_cast<float>(row_input[i]);
    }
    sum = warp_reduce_sum(sum);
    sum = __shfl_sync(0xffffffff, sum, 0);
    float mean = sum / hidden_size;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = static_cast<float>(row_input[i]) - mean;
        var_sum += diff * diff;
    }
    var_sum = warp_reduce_sum(var_sum);
    var_sum = __shfl_sync(0xffffffff, var_sum, 0);
    float inv_std = rsqrtf(var_sum / hidden_size + eps);

    // Normalize and apply affine transform
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (static_cast<float>(row_input[i]) - mean) * inv_std;
        row_output[i] = static_cast<T>(normalized * static_cast<float>(gamma[i]) +
                                       static_cast<float>(beta[i]));
    }
}

template <>
void layer_norm<float>(const float* input, const float* gamma, const float* beta,
                       float* output, int batch, int hidden_size,
                       float eps, cudaStream_t stream) {
    int block_size = 256;
    layer_norm_kernel<float><<<batch, block_size, 0, stream>>>(
        input, gamma, beta, output, hidden_size, eps);
    CUDA_CHECK_LAST();
}

template <>
void layer_norm<__half>(const __half* input, const __half* gamma, const __half* beta,
                        __half* output, int batch, int hidden_size,
                        float eps, cudaStream_t stream) {
    int block_size = 256;
    layer_norm_kernel<__half><<<batch, block_size, 0, stream>>>(
        input, gamma, beta, output, hidden_size, eps);
    CUDA_CHECK_LAST();
}

} // namespace hpc::reduction
