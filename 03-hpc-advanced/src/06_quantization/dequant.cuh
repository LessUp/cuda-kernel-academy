#pragma once

#include <cuda_runtime.h>

namespace hpc::quantization {

template <typename T>
void dequantize_weight(const int8_t* quantized, const float* scale,
                       T* output, int rows, int cols,
                       cudaStream_t stream = nullptr);

} // namespace hpc::quantization
