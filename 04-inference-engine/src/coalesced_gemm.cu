#include "kernels.cuh"

#include "tensorcraft/kernels/gemm.hpp"

namespace mini_inference {

// ============================================================================
// Level 3: Coalesced Memory Access GEMM
// Optimizes memory access patterns for better bandwidth utilization
// ============================================================================

#define TILE_SIZE 32

__global__ void coalesced_gemm(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    // Use padding to avoid bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Coalesced load from A: threads in same warp access consecutive addresses
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Coalesced load from B: threads in same warp access consecutive addresses
        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute with unrolling for better ILP
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_coalesced_gemm(const float* A, const float* B, float* C,
                           int M, int N, int K, cudaStream_t stream) {
    tensorcraft::kernels::launch_gemm(
        A, B, C, M, N, K,
        1.0f, 0.0f,
        tensorcraft::kernels::GemmVersion::Tiled,
        stream);
}

#undef TILE_SIZE

} // namespace mini_inference
