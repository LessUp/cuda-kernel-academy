#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <concepts>

namespace hpc::reduction {

enum class SoftmaxOpt {
    Naive,         // Two-pass with global atomics
    WarpShuffle,   // Warp-level reduction
    OnlineSoftmax, // Single-pass online algorithm
    Fused          // Fused with L2 cache persistence
};

template <typename T, SoftmaxOpt Opt = SoftmaxOpt::OnlineSoftmax>
    requires std::is_same_v<T, float> || std::is_same_v<T, __half>
void softmax(const T* input, T* output, int batch, int seq_len,
             cudaStream_t stream = nullptr);

} // namespace hpc::reduction
