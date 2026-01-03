#pragma once

#include <cuda_runtime.h>

namespace hpc::cuda13 {

template <typename T>
void tma_copy_2d(const T* src, T* dst,
                 int rows, int cols,
                 cudaStream_t stream = nullptr);

} // namespace hpc::cuda13
