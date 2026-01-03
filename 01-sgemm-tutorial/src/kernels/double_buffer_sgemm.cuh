#pragma once

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

/**
 * Double Buffer SGEMM Kernel
 * 
 * This implementation uses double buffering (software pipelining) to overlap
 * computation with memory access, hiding global memory latency.
 * 
 * ============================================================================
 * Double Buffering Concept:
 * ============================================================================
 * 
 * Without double buffering:
 *   Load tile 0 -> Compute tile 0 -> Load tile 1 -> Compute tile 1 -> ...
 *   [Memory]      [Compute]         [Memory]      [Compute]
 * 
 * With double buffering:
 *   Load tile 0 -> Load tile 1    -> Load tile 2    -> ...
 *                  Compute tile 0 -> Compute tile 1 -> ...
 *   [Memory]      [Memory+Compute]  [Memory+Compute]
 * 
 * By using two sets of buffers, we can load the next tile while computing
 * on the current tile, effectively hiding memory latency.
 * 
 * ============================================================================
 * Implementation Details:
 * ============================================================================
 * 
 * 1. Allocate two sets of shared memory buffers (As[2], Bs[2])
 * 2. Pre-load the first tile into buffer 0
 * 3. Main loop:
 *    - While computing on buffer[read_idx], load next tile into buffer[write_idx]
 *    - Swap buffer indices
 * 4. Handle the last tile (no prefetch needed)
 * 
 * C = A * B
 * A: M x K (row-major)
 * B: K x N (row-major)
 * C: M x N (row-major)
 */
template<int TILE_SIZE>
__global__ void double_buffer_sgemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // Double buffers with padding to avoid bank conflicts
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Buffer indices for double buffering
    int writeBuffer = 0;
    int readBuffer = 0;
    
    // ========================================================================
    // Pre-load first tile into buffer 0
    // ========================================================================
    {
        int aCol = tx;  // t = 0
        if (row < M && aCol < K) {
            As[0][ty][tx] = A[row * K + aCol];
        } else {
            As[0][ty][tx] = 0.0f;
        }
        
        int bRow = ty;  // t = 0
        if (bRow < K && col < N) {
            Bs[0][ty][tx] = B[bRow * N + col];
        } else {
            Bs[0][ty][tx] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // ========================================================================
    // Main loop with double buffering
    // ========================================================================
    for (int t = 0; t < numTiles; ++t) {
        readBuffer = writeBuffer;
        writeBuffer = 1 - writeBuffer;
        
        // Prefetch next tile (if not the last iteration)
        if (t + 1 < numTiles) {
            int nextT = t + 1;
            
            int aCol = nextT * TILE_SIZE + tx;
            if (row < M && aCol < K) {
                As[writeBuffer][ty][tx] = A[row * K + aCol];
            } else {
                As[writeBuffer][ty][tx] = 0.0f;
            }
            
            int bRow = nextT * TILE_SIZE + ty;
            if (bRow < K && col < N) {
                Bs[writeBuffer][ty][tx] = B[bRow * N + col];
            } else {
                Bs[writeBuffer][ty][tx] = 0.0f;
            }
        }
        
        // Compute on current buffer while prefetching
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[readBuffer][ty][k] * Bs[readBuffer][k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Launch wrapper for double buffer SGEMM kernel
 */
template<int TILE_SIZE = 32>
void launch_double_buffer_sgemm(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N,
    cudaStream_t stream = 0
) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );
    
    double_buffer_sgemm_kernel<TILE_SIZE><<<gridDim, blockDim, 0, stream>>>(
        A, B, C, M, K, N
    );
    
    CUDA_CHECK(cudaGetLastError());
}

/**
 * Advanced Double Buffer with Register Tiling
 * 
 * Further optimization: each thread computes multiple output elements
 * using register-level tiling to increase arithmetic intensity.
 */
template<int TILE_SIZE, int THREAD_TILE_M = 4, int THREAD_TILE_N = 4>
__global__ void double_buffer_register_tiled_sgemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // Each thread computes a THREAD_TILE_M x THREAD_TILE_N sub-matrix
    // Block computes TILE_SIZE x TILE_SIZE output
    constexpr int THREADS_PER_BLOCK_X = TILE_SIZE / THREAD_TILE_N;
    constexpr int THREADS_PER_BLOCK_Y = TILE_SIZE / THREAD_TILE_M;
    
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Thread's starting position in the output tile
    int threadRow = ty * THREAD_TILE_M;
    int threadCol = tx * THREAD_TILE_N;
    
    // Global starting position
    int globalRowStart = by * TILE_SIZE + threadRow;
    int globalColStart = bx * TILE_SIZE + threadCol;
    
    // Register accumulators for the thread's sub-matrix
    float regC[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};
    
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    int writeBuffer = 0;
    
    // Pre-load first tile (collaborative loading)
    // Each thread loads multiple elements
    for (int i = ty; i < TILE_SIZE; i += THREADS_PER_BLOCK_Y) {
        for (int j = tx; j < TILE_SIZE; j += THREADS_PER_BLOCK_X) {
            int globalRow = by * TILE_SIZE + i;
            int globalCol = j;
            
            if (globalRow < M && globalCol < K) {
                As[0][i][j] = A[globalRow * K + globalCol];
            } else {
                As[0][i][j] = 0.0f;
            }
            
            int bGlobalRow = i;
            int bGlobalCol = bx * TILE_SIZE + j;
            
            if (bGlobalRow < K && bGlobalCol < N) {
                Bs[0][i][j] = B[bGlobalRow * N + bGlobalCol];
            } else {
                Bs[0][i][j] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    for (int t = 0; t < numTiles; ++t) {
        int readBuffer = writeBuffer;
        writeBuffer = 1 - writeBuffer;
        
        // Prefetch next tile
        if (t + 1 < numTiles) {
            int nextT = t + 1;
            for (int i = ty; i < TILE_SIZE; i += THREADS_PER_BLOCK_Y) {
                for (int j = tx; j < TILE_SIZE; j += THREADS_PER_BLOCK_X) {
                    int globalRow = by * TILE_SIZE + i;
                    int globalCol = nextT * TILE_SIZE + j;
                    
                    if (globalRow < M && globalCol < K) {
                        As[writeBuffer][i][j] = A[globalRow * K + globalCol];
                    } else {
                        As[writeBuffer][i][j] = 0.0f;
                    }
                    
                    int bGlobalRow = nextT * TILE_SIZE + i;
                    int bGlobalCol = bx * TILE_SIZE + j;
                    
                    if (bGlobalRow < K && bGlobalCol < N) {
                        Bs[writeBuffer][i][j] = B[bGlobalRow * N + bGlobalCol];
                    } else {
                        Bs[writeBuffer][i][j] = 0.0f;
                    }
                }
            }
        }
        
        // Compute using register tiling
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            // Load A values into registers
            float regA[THREAD_TILE_M];
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                regA[i] = As[readBuffer][threadRow + i][k];
            }
            
            // Load B values into registers
            float regB[THREAD_TILE_N];
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j) {
                regB[j] = Bs[readBuffer][k][threadCol + j];
            }
            
            // Outer product
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    regC[i][j] += regA[i] * regB[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            int globalRow = globalRowStart + i;
            int globalCol = globalColStart + j;
            if (globalRow < M && globalCol < N) {
                C[globalRow * N + globalCol] = regC[i][j];
            }
        }
    }
}

template<int TILE_SIZE = 64, int THREAD_TILE_M = 4, int THREAD_TILE_N = 4>
void launch_double_buffer_register_tiled_sgemm(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N,
    cudaStream_t stream = 0
) {
    constexpr int THREADS_X = TILE_SIZE / THREAD_TILE_N;
    constexpr int THREADS_Y = TILE_SIZE / THREAD_TILE_M;
    
    dim3 blockDim(THREADS_X, THREADS_Y);
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );
    
    double_buffer_register_tiled_sgemm_kernel<TILE_SIZE, THREAD_TILE_M, THREAD_TILE_N>
        <<<gridDim, blockDim, 0, stream>>>(A, B, C, M, K, N);
    
    CUDA_CHECK(cudaGetLastError());
}
