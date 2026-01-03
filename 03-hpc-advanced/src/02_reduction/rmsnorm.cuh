#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <concepts>

namespace hpc::reduction {

template <typename T>
    requires std::is_same_v<T, float> || std::is_same_v<T, __half>
void rms_norm(const T* input, const T* gamma,
              T* output, int batch, int hidden_size,
              float eps = 1e-5f, cudaStream_t stream = nullptr);

} // namespace hpc::reduction
