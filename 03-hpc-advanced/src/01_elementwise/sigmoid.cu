#include "sigmoid.cuh"
#include "../common/cuda_check.cuh"
#include <cmath>

namespace hpc::elementwise {

template <typename T>
__global__ void sigmoid_naive_kernel(const T* __restrict__ input,
                                      T* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = static_cast<float>(input[idx]);
        output[idx] = static_cast<T>(1.0f / (1.0f + expf(-val)));
    }
}

template <typename T>
__global__ void sigmoid_grid_stride_kernel(const T* __restrict__ input,
                                            T* __restrict__ output, size_t n) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += blockDim.x * gridDim.x) {
        float val = static_cast<float>(input[idx]);
        output[idx] = static_cast<T>(1.0f / (1.0f + expf(-val)));
    }
}

template <>
void sigmoid<float, OptLevel::Naive>(const float* input, float* output,
                                      size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    sigmoid_naive_kernel<float><<<grid_size, block_size, 0, stream>>>(input, output, n);
    CUDA_CHECK_LAST();
}

template <>
void sigmoid<float, OptLevel::GridStride>(const float* input, float* output,
                                           size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = std::min(static_cast<int>((n + block_size - 1) / block_size), 1024);
    sigmoid_grid_stride_kernel<float><<<grid_size, block_size, 0, stream>>>(input, output, n);
    CUDA_CHECK_LAST();
}

template <>
void sigmoid<__half, OptLevel::GridStride>(const __half* input, __half* output,
                                            size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = std::min(static_cast<int>((n + block_size - 1) / block_size), 1024);
    sigmoid_grid_stride_kernel<__half><<<grid_size, block_size, 0, stream>>>(input, output, n);
    CUDA_CHECK_LAST();
}

} // namespace hpc::elementwise
