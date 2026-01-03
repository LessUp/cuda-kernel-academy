#include "relu.cuh"
#include "../common/cuda_check.cuh"
#include "../common/launch.cuh"

namespace hpc::elementwise {

// Naive ReLU kernel
template <typename T>
__global__ void relu_naive_kernel(const T* __restrict__ input,
                                   T* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T val = input[idx];
        output[idx] = val > T(0) ? val : T(0);
    }
}

// Vectorized ReLU kernel (float4)
__global__ void relu_vectorized_kernel(const float* __restrict__ input,
                                        float* __restrict__ output, size_t n) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        float4 val = reinterpret_cast<const float4*>(input)[idx / 4];
        val.x = val.x > 0.0f ? val.x : 0.0f;
        val.y = val.y > 0.0f ? val.y : 0.0f;
        val.z = val.z > 0.0f ? val.z : 0.0f;
        val.w = val.w > 0.0f ? val.w : 0.0f;
        reinterpret_cast<float4*>(output)[idx / 4] = val;
    } else {
        // Handle remaining elements
        for (size_t i = idx; i < n; ++i) {
            float val = input[i];
            output[i] = val > 0.0f ? val : 0.0f;
        }
    }
}

// Grid stride loop ReLU kernel
template <typename T>
__global__ void relu_grid_stride_kernel(const T* __restrict__ input,
                                         T* __restrict__ output, size_t n) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += blockDim.x * gridDim.x) {
        T val = input[idx];
        output[idx] = val > T(0) ? val : T(0);
    }
}

// Template specializations
template <>
void relu<float, OptLevel::Naive>(const float* input, float* output, size_t n,
                                   cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    relu_naive_kernel<float><<<grid_size, block_size, 0, stream>>>(input, output, n);
    CUDA_CHECK_LAST();
}

template <>
void relu<float, OptLevel::Vectorized>(const float* input, float* output,
                                        size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n / 4 + block_size - 1) / block_size);
    relu_vectorized_kernel<<<grid_size, block_size, 0, stream>>>(input, output, n);
    CUDA_CHECK_LAST();
}

template <>
void relu<float, OptLevel::GridStride>(const float* input, float* output,
                                        size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = std::min(static_cast<int>((n + block_size - 1) / block_size), 1024);
    relu_grid_stride_kernel<float><<<grid_size, block_size, 0, stream>>>(input, output, n);
    CUDA_CHECK_LAST();
}

template <>
void relu<__half, OptLevel::Naive>(const __half* input, __half* output,
                                    size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    relu_naive_kernel<__half><<<grid_size, block_size, 0, stream>>>(input, output, n);
    CUDA_CHECK_LAST();
}

template <>
void relu<__half, OptLevel::GridStride>(const __half* input, __half* output,
                                         size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = std::min(static_cast<int>((n + block_size - 1) / block_size), 1024);
    relu_grid_stride_kernel<__half><<<grid_size, block_size, 0, stream>>>(input, output, n);
    CUDA_CHECK_LAST();
}

} // namespace hpc::elementwise
