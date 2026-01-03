#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <concepts>

namespace hpc::elementwise {

enum class TransposeOpt {
    Naive,           // Direct read-row write-col
    SharedMemory,    // Use shared memory
    SharedMemPadded  // Shared memory with padding to avoid bank conflict
};

template <typename T, TransposeOpt Opt = TransposeOpt::SharedMemPadded>
    requires std::is_same_v<T, float> || std::is_same_v<T, __half>
void transpose(const T* input, T* output, int rows, int cols,
               cudaStream_t stream = nullptr);

} // namespace hpc::elementwise
