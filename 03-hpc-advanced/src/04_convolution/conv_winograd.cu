#include "conv_winograd.cuh"
#include "../common/cuda_check.cuh"

namespace hpc::convolution {

// Winograd F(2x2, 3x3) transformation matrices
// TODO: Implement full Winograd convolution

template <>
void conv2d_winograd<float>(const float* input, const float* weight, float* output,
                            int batch, int in_channels, int out_channels,
                            int height, int width, cudaStream_t stream) {
    // Placeholder - full implementation requires Winograd transforms
    // For now, fall back to implicit GEMM
}

} // namespace hpc::convolution
