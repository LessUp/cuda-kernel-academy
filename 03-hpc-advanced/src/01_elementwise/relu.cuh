#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <concepts>

namespace hpc::elementwise {

// Optimization levels for elementwise operations
enum class OptLevel {
    Naive,       // Basic implementation
    Vectorized,  // float4 load/store
    GridStride   // Grid stride loop
};

// ReLU kernel interface
template <typename T, OptLevel Level = OptLevel::GridStride>
    requires std::is_same_v<T, float> || std::is_same_v<T, __half>
void relu(const T* input, T* output, size_t n, cudaStream_t stream = nullptr);

} // namespace hpc::elementwise
