#pragma once

#include <cuda_runtime.h>

namespace hpc::attention {

template <typename T>
void topk(const T* input, T* output, int* indices,
          int batch, int n, int k,
          cudaStream_t stream = nullptr);

} // namespace hpc::attention
