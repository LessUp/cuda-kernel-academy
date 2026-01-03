#pragma once

#include <cuda_runtime.h>

namespace hpc::quantization {

void fp8_scale(const float* input, float* output, float scale, int n,
               cudaStream_t stream = nullptr);

} // namespace hpc::quantization
