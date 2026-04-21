#include "kernels.cuh"

#include "tensorcraft/kernels/gemm.hpp"

namespace mini_inference {

// ============================================================================
// Level 4: Double Buffered GEMM
// Uses two shared memory buffers to overlap computation with data loading
// ============================================================================

#define TILE_SIZE 32

__global__ void double_buffer_gemm(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    // Double buffers for A and B tiles
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    int write_stage = 0;

    // Load first tile
    {
        int a_col = tx;
        if (row < M && a_col < K) {
            As[0][ty][tx] = A[row * K + a_col];
        } else {
            As[0][ty][tx] = 0.0f;
        }

        int b_row = ty;
        if (b_row < K && col < N) {
            Bs[0][ty][tx] = B[b_row * N + col];
        } else {
            Bs[0][ty][tx] = 0.0f;
        }
    }

    __syncthreads();

    // Main loop with double buffering
    for (int t = 0; t < num_tiles; t++) {
        int read_stage = write_stage;
        write_stage = 1 - write_stage;

        // Prefetch next tile (if not last iteration)
        if (t + 1 < num_tiles) {
            int next_tile = t + 1;
            int a_col = next_tile * TILE_SIZE + tx;
            if (row < M && a_col < K) {
                As[write_stage][ty][tx] = A[row * K + a_col];
            } else {
                As[write_stage][ty][tx] = 0.0f;
            }

            int b_row = next_tile * TILE_SIZE + ty;
            if (b_row < K && col < N) {
                Bs[write_stage][ty][tx] = B[b_row * N + col];
            } else {
                Bs[write_stage][ty][tx] = 0.0f;
            }
        }

        // Compute on current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[read_stage][ty][k] * Bs[read_stage][k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_double_buffer_gemm(const float* A, const float* B, float* C,
                               int M, int N, int K, cudaStream_t stream) {
    tensorcraft::kernels::launch_gemm(
        A, B, C, M, N, K,
        1.0f, 0.0f,
        tensorcraft::kernels::GemmVersion::DoubleBuffer,
        stream);
}

#undef TILE_SIZE

} // namespace mini_inference
