#include "transpose.cuh"
#include "../common/cuda_check.cuh"

namespace hpc::elementwise {

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

// Naive transpose: read row, write column (non-coalesced writes)
template <typename T>
__global__ void transpose_naive_kernel(const T* __restrict__ input,
                                        T* __restrict__ output,
                                        int rows, int cols) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && (y + j) < rows) {
            output[x * rows + (y + j)] = input[(y + j) * cols + x];
        }
    }
}

// Shared memory transpose: coalesced reads and writes
template <typename T>
__global__ void transpose_shared_kernel(const T* __restrict__ input,
                                         T* __restrict__ output,
                                         int rows, int cols) {
    __shared__ T tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile into shared memory (coalesced read)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
        }
    }

    __syncthreads();

    // Write transposed tile (coalesced write)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < rows && (y + j) < cols) {
            output[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Shared memory with padding to avoid bank conflicts
template <typename T>
__global__ void transpose_shared_padded_kernel(const T* __restrict__ input,
                                                T* __restrict__ output,
                                                int rows, int cols) {
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];  // +1 padding to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile into shared memory (coalesced read)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
        }
    }

    __syncthreads();

    // Write transposed tile (coalesced write)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < rows && (y + j) < cols) {
            output[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

template <>
void transpose<float, TransposeOpt::Naive>(const float* input, float* output,
                                            int rows, int cols, cudaStream_t stream) {
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);
    transpose_naive_kernel<float><<<grid, block, 0, stream>>>(input, output, rows, cols);
    CUDA_CHECK_LAST();
}

template <>
void transpose<float, TransposeOpt::SharedMemory>(const float* input, float* output,
                                                   int rows, int cols, cudaStream_t stream) {
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);
    transpose_shared_kernel<float><<<grid, block, 0, stream>>>(input, output, rows, cols);
    CUDA_CHECK_LAST();
}

template <>
void transpose<float, TransposeOpt::SharedMemPadded>(const float* input, float* output,
                                                      int rows, int cols, cudaStream_t stream) {
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);
    transpose_shared_padded_kernel<float><<<grid, block, 0, stream>>>(input, output, rows, cols);
    CUDA_CHECK_LAST();
}

template <>
void transpose<__half, TransposeOpt::SharedMemPadded>(const __half* input, __half* output,
                                                       int rows, int cols, cudaStream_t stream) {
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);
    transpose_shared_padded_kernel<__half><<<grid, block, 0, stream>>>(input, output, rows, cols);
    CUDA_CHECK_LAST();
}

} // namespace hpc::elementwise
