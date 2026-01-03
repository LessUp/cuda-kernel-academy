#pragma once

#include <cuda_runtime.h>

namespace hpc::convolution {

template <typename T>
void conv2d_winograd(const T* input, const T* weight, T* output,
                     int batch, int in_channels, int out_channels,
                     int height, int width,
                     cudaStream_t stream = nullptr);

} // namespace hpc::convolution
