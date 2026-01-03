#pragma once

#include <cuda_runtime.h>

namespace hpc::quantization {

void quantize_int8(const float* input, int8_t* output, float* scale,
                   int rows, int cols, cudaStream_t stream = nullptr);

void dequantize_int8(const int8_t* input, const float* scale,
                     float* output, int rows, int cols,
                     cudaStream_t stream = nullptr);

} // namespace hpc::quantization
