#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace hpc::attention {

template <typename T>
void apply_rope(T* query, T* key,
                int batch, int num_heads, int seq_len, int head_dim,
                const float* cos_cache, const float* sin_cache,
                cudaStream_t stream = nullptr);

} // namespace hpc::attention
