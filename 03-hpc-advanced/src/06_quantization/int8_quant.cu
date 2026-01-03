#include "int8_quant.cuh"
#include "../common/cuda_check.cuh"
#include <cfloat>

namespace hpc::quantization {

__global__ void compute_scale_kernel(const float* __restrict__ input,
                                      float* __restrict__ scale,
                                      int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    const float* row_input = input + row * cols;
    float max_abs = 0.0f;
    
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        max_abs = fmaxf(max_abs, fabsf(row_input[i]));
    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
    }
    
    if (threadIdx.x == 0) {
        scale[row] = max_abs / 127.0f;
    }
}

__global__ void quantize_kernel(const float* __restrict__ input,
                                 int8_t* __restrict__ output,
                                 const float* __restrict__ scale,
                                 int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    if (idx < total) {
        int row = idx / cols;
        float inv_scale = 1.0f / scale[row];
        float val = input[idx] * inv_scale;
        val = fminf(fmaxf(val, -127.0f), 127.0f);
        output[idx] = static_cast<int8_t>(roundf(val));
    }
}

void quantize_int8(const float* input, int8_t* output, float* scale,
                   int rows, int cols, cudaStream_t stream) {
    compute_scale_kernel<<<rows, 256, 0, stream>>>(input, scale, rows, cols);
    
    int total = rows * cols;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    quantize_kernel<<<grid_size, block_size, 0, stream>>>(input, output, scale, rows, cols);
    CUDA_CHECK_LAST();
}

__global__ void dequantize_int8_kernel(const int8_t* __restrict__ input,
                                        const float* __restrict__ scale,
                                        float* __restrict__ output,
                                        int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    if (idx < total) {
        int row = idx / cols;
        output[idx] = static_cast<float>(input[idx]) * scale[row];
    }
}

void dequantize_int8(const int8_t* input, const float* scale,
                     float* output, int rows, int cols, cudaStream_t stream) {
    int total = rows * cols;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    dequantize_int8_kernel<<<grid_size, block_size, 0, stream>>>(
        input, scale, output, rows, cols);
    CUDA_CHECK_LAST();
}

} // namespace hpc::quantization
