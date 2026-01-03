#pragma once

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

/**
 * Tiled SGEMM Kernel using Shared Memory
 * 
 * This implementation uses shared memory tiling to reduce global memory accesses.
 * Each thread block loads tiles of A and B into shared memory, then computes
 * partial products using the cached data.
 * 
 * Performance Improvements over Naive:
 * 1. Coalesced global memory access when loading tiles
 * 2. Data reuse: each element loaded from global memory is used TILE_SIZE times
 * 3. Higher arithmetic intensity
 * 
 * Memory Access Pattern:
 * - Global memory reads reduced by factor of TILE_SIZE
 * - Shared memory provides low-latency access for repeated reads
 * 
 * C = A * B
 * A: M x K (row-major)
 * B: K x N (row-major)
 * C: M x N (row-major)
 */
template<int TILE_SIZE>
__global__ void tiled_sgemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Block and thread indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate global row and column for this thread
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    // Accumulator for the dot product
    float sum = 0.0f;
    
    // Number of tiles needed to cover K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Iterate over all tiles
    for (int t = 0; t < numTiles; ++t) {
        // Load tile of A into shared memory
        // Each thread loads one element
        int aCol = t * TILE_SIZE + tx;
        if (row < M && aCol < K) {
            As[ty][tx] = A[row * K + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile of B into shared memory
        // Each thread loads one element
        int bRow = t * TILE_SIZE + ty;
        if (bRow < K && col < N) {
            Bs[ty][tx] = B[bRow * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // Synchronize to ensure all threads have loaded their data
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        // Synchronize before loading next tile
        // This prevents threads from overwriting shared memory
        // while other threads are still reading
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Launch wrapper for tiled SGEMM kernel
 * 
 * @tparam TILE_SIZE Size of the tile (default 32)
 * @param A Device pointer to matrix A (M x K)
 * @param B Device pointer to matrix B (K x N)
 * @param C Device pointer to output matrix C (M x N)
 * @param M Number of rows in A and C
 * @param K Number of columns in A, rows in B
 * @param N Number of columns in B and C
 * @param stream CUDA stream (default: 0)
 */
template<int TILE_SIZE = 32>
void launch_tiled_sgemm(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N,
    cudaStream_t stream = 0
) {
    // Block size matches tile size
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    
    // Grid covers the output matrix
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );
    
    // Launch kernel
    tiled_sgemm_kernel<TILE_SIZE><<<gridDim, blockDim, 0, stream>>>(
        A, B, C, M, K, N
    );
    
    CUDA_CHECK(cudaGetLastError());
}

/**
 * Tiled SGEMM with alpha and beta scaling
 * C = alpha * A * B + beta * C
 */
template<int TILE_SIZE>
__global__ void tiled_sgemm_kernel_scaled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N,
    float alpha, float beta
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_SIZE + tx;
        if (row < M && aCol < K) {
            As[ty][tx] = A[row * K + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        int bRow = t * TILE_SIZE + ty;
        if (bRow < K && col < N) {
            Bs[ty][tx] = B[bRow * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

template<int TILE_SIZE = 32>
void launch_tiled_sgemm_scaled(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N,
    float alpha, float beta,
    cudaStream_t stream = 0
) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );
    
    tiled_sgemm_kernel_scaled<TILE_SIZE><<<gridDim, blockDim, 0, stream>>>(
        A, B, C, M, K, N, alpha, beta
    );
    
    CUDA_CHECK(cudaGetLastError());
}
