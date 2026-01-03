#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <concepts>
#include "relu.cuh"  // For OptLevel enum

namespace hpc::elementwise {

template <typename T, OptLevel Level = OptLevel::GridStride>
    requires std::is_same_v<T, float> || std::is_same_v<T, __half>
void vector_add(const T* a, const T* b, T* c, size_t n, cudaStream_t stream = nullptr);

} // namespace hpc::elementwise
