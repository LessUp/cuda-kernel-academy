#include "dequant.cuh"
#include "../common/cuda_check.cuh"

namespace hpc::quantization {

template <typename T>
__global__ void dequantize_kernel(const int8_t* __restrict__ quantized,
                                   const float* __restrict__ scale,
                                   T* __restrict__ output,
                                   int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    if (idx < total) {
        int row = idx / cols;
        output[idx] = static_cast<T>(static_cast<float>(quantized[idx]) * scale[row]);
    }
}

template <>
void dequantize_weight<float>(const int8_t* quantized, const float* scale,
                              float* output, int rows, int cols,
                              cudaStream_t stream) {
    int total = rows * cols;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    dequantize_kernel<float><<<grid_size, block_size, 0, stream>>>(
        quantized, scale, output, rows, cols);
    CUDA_CHECK_LAST();
}

} // namespace hpc::quantization
