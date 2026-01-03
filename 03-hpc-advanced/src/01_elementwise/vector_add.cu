#include "vector_add.cuh"
#include "../common/cuda_check.cuh"

namespace hpc::elementwise {

template <typename T>
__global__ void vector_add_naive_kernel(const T* __restrict__ a,
                                         const T* __restrict__ b,
                                         T* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_add_vectorized_kernel(const float* __restrict__ a,
                                              const float* __restrict__ b,
                                              float* __restrict__ c, size_t n) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        float4 va = reinterpret_cast<const float4*>(a)[idx / 4];
        float4 vb = reinterpret_cast<const float4*>(b)[idx / 4];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        reinterpret_cast<float4*>(c)[idx / 4] = vc;
    } else {
        for (size_t i = idx; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    }
}

template <typename T>
__global__ void vector_add_grid_stride_kernel(const T* __restrict__ a,
                                               const T* __restrict__ b,
                                               T* __restrict__ c, size_t n) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += blockDim.x * gridDim.x) {
        c[idx] = a[idx] + b[idx];
    }
}

template <>
void vector_add<float, OptLevel::Naive>(const float* a, const float* b,
                                         float* c, size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    vector_add_naive_kernel<float><<<grid_size, block_size, 0, stream>>>(a, b, c, n);
    CUDA_CHECK_LAST();
}

template <>
void vector_add<float, OptLevel::Vectorized>(const float* a, const float* b,
                                              float* c, size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n / 4 + block_size - 1) / block_size);
    vector_add_vectorized_kernel<<<grid_size, block_size, 0, stream>>>(a, b, c, n);
    CUDA_CHECK_LAST();
}

template <>
void vector_add<float, OptLevel::GridStride>(const float* a, const float* b,
                                              float* c, size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = std::min(static_cast<int>((n + block_size - 1) / block_size), 1024);
    vector_add_grid_stride_kernel<float><<<grid_size, block_size, 0, stream>>>(a, b, c, n);
    CUDA_CHECK_LAST();
}

template <>
void vector_add<__half, OptLevel::GridStride>(const __half* a, const __half* b,
                                               __half* c, size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = std::min(static_cast<int>((n + block_size - 1) / block_size), 1024);
    vector_add_grid_stride_kernel<__half><<<grid_size, block_size, 0, stream>>>(a, b, c, n);
    CUDA_CHECK_LAST();
}

} // namespace hpc::elementwise
