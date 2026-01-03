#include "softmax.cuh"
#include "../common/cuda_check.cuh"
#include <cfloat>
#include <cmath>

namespace hpc::reduction {

// Warp-level reduction using shuffle
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Online softmax kernel (single pass)
template <typename T>
__global__ void softmax_online_kernel(const T* __restrict__ input,
                                       T* __restrict__ output,
                                       int batch, int seq_len) {
    int row = blockIdx.x;
    if (row >= batch) return;

    const T* row_input = input + row * seq_len;
    T* row_output = output + row * seq_len;

    // Online algorithm: compute max and sum in single pass
    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;

    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float val = static_cast<float>(row_input[i]);
        float old_max = max_val;
        max_val = fmaxf(max_val, val);
        sum_exp = sum_exp * expf(old_max - max_val) + expf(val - max_val);
    }

    // Warp reduction for max
    max_val = warp_reduce_max(max_val);
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    // Recompute sum with correct max
    sum_exp = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        sum_exp += expf(static_cast<float>(row_input[i]) - max_val);
    }
    sum_exp = warp_reduce_sum(sum_exp);
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

    // Compute softmax output
    float inv_sum = 1.0f / sum_exp;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        row_output[i] = static_cast<T>(expf(static_cast<float>(row_input[i]) - max_val) * inv_sum);
    }
}

template <>
void softmax<float, SoftmaxOpt::OnlineSoftmax>(const float* input, float* output,
                                                int batch, int seq_len, cudaStream_t stream) {
    int block_size = 256;
    softmax_online_kernel<float><<<batch, block_size, 0, stream>>>(input, output, batch, seq_len);
    CUDA_CHECK_LAST();
}

template <>
void softmax<__half, SoftmaxOpt::OnlineSoftmax>(const __half* input, __half* output,
                                                 int batch, int seq_len, cudaStream_t stream) {
    int block_size = 256;
    softmax_online_kernel<__half><<<batch, block_size, 0, stream>>>(input, output, batch, seq_len);
    CUDA_CHECK_LAST();
}

} // namespace hpc::reduction
