#include "rope.cuh"
#include "../common/cuda_check.cuh"

namespace hpc::attention {

template <typename T>
__global__ void apply_rope_kernel(T* __restrict__ query,
                                   T* __restrict__ key,
                                   const float* __restrict__ cos_cache,
                                   const float* __restrict__ sin_cache,
                                   int batch, int num_heads,
                                   int seq_len, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_heads * seq_len * (head_dim / 2);
    
    if (idx >= total) return;
    
    int half_dim = head_dim / 2;
    int d = idx % half_dim;
    int s = (idx / half_dim) % seq_len;
    int h = (idx / (half_dim * seq_len)) % num_heads;
    int b = idx / (half_dim * seq_len * num_heads);
    
    int base_idx = (b * num_heads * seq_len + h * seq_len + s) * head_dim;
    int cos_sin_idx = s * half_dim + d;
    
    float cos_val = cos_cache[cos_sin_idx];
    float sin_val = sin_cache[cos_sin_idx];
    
    // Apply rotation to query
    float q0 = static_cast<float>(query[base_idx + d]);
    float q1 = static_cast<float>(query[base_idx + d + half_dim]);
    query[base_idx + d] = static_cast<T>(q0 * cos_val - q1 * sin_val);
    query[base_idx + d + half_dim] = static_cast<T>(q0 * sin_val + q1 * cos_val);
    
    // Apply rotation to key
    float k0 = static_cast<float>(key[base_idx + d]);
    float k1 = static_cast<float>(key[base_idx + d + half_dim]);
    key[base_idx + d] = static_cast<T>(k0 * cos_val - k1 * sin_val);
    key[base_idx + d + half_dim] = static_cast<T>(k0 * sin_val + k1 * cos_val);
}

template <>
void apply_rope<float>(float* query, float* key,
                       int batch, int num_heads, int seq_len, int head_dim,
                       const float* cos_cache, const float* sin_cache,
                       cudaStream_t stream) {
    int total = batch * num_heads * seq_len * (head_dim / 2);
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    apply_rope_kernel<float><<<grid_size, block_size, 0, stream>>>(
        query, key, cos_cache, sin_cache,
        batch, num_heads, seq_len, head_dim);
    CUDA_CHECK_LAST();
}

template <>
void apply_rope<__half>(__half* query, __half* key,
                        int batch, int num_heads, int seq_len, int head_dim,
                        const float* cos_cache, const float* sin_cache,
                        cudaStream_t stream) {
    int total = batch * num_heads * seq_len * (head_dim / 2);
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    apply_rope_kernel<__half><<<grid_size, block_size, 0, stream>>>(
        query, key, cos_cache, sin_cache,
        batch, num_heads, seq_len, head_dim);
    CUDA_CHECK_LAST();
}

} // namespace hpc::attention
