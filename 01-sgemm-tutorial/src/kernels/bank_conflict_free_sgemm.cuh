#pragma once

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

/**
 * Bank Conflict Free SGEMM Kernel
 * 
 * This implementation eliminates shared memory bank conflicts by adding padding
 * to the shared memory arrays.
 * 
 * ============================================================================
 * Bank Conflict Explanation:
 * ============================================================================
 * 
 * Shared memory is divided into 32 banks (on modern GPUs).
 * Each bank is 4 bytes wide (one float).
 * Consecutive 4-byte words map to consecutive banks.
 * 
 * Bank assignment: bank_id = (address / 4) % 32
 * 
 * For a 32x32 array stored row-major:
 *   - Element [i][j] is at address: (i * 32 + j) * 4
 *   - Bank: (i * 32 + j) % 32 = j (since 32 % 32 = 0)
 * 
 * Problem: When threads in a warp access column j of the array,
 * they all access the same bank j, causing a 32-way bank conflict!
 * 
 * Solution: Add 1 element of padding per row.
 *   - Element [i][j] is now at address: (i * 33 + j) * 4
 *   - Bank: (i * 33 + j) % 32 = (i + j) % 32
 *   - Now threads accessing column j get different banks!
 * 
 * ============================================================================
 * 
 * C = A * B
 * A: M x K (row-major)
 * B: K x N (row-major)
 * C: M x N (row-major)
 */
template<int TILE_SIZE>
__global__ void bank_conflict_free_sgemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // Shared memory with padding to avoid bank conflicts
    // Adding 1 to the second dimension shifts each row by 1 bank
    // This ensures column accesses hit different banks
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 padding
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];  // +1 padding
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        // Load tile of A into shared memory (coalesced access)
        int aCol = t * TILE_SIZE + tx;
        if (row < M && aCol < K) {
            As[ty][tx] = A[row * K + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile of B into shared memory (coalesced access)
        int bRow = t * TILE_SIZE + ty;
        if (bRow < K && col < N) {
            Bs[ty][tx] = B[bRow * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        // Access pattern: As[ty][k] - row access (no conflict)
        //                 Bs[k][tx] - column access (no conflict due to padding!)
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Launch wrapper for bank conflict free SGEMM kernel
 */
template<int TILE_SIZE = 32>
void launch_bank_conflict_free_sgemm(
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
    
    bank_conflict_free_sgemm_kernel<TILE_SIZE><<<gridDim, blockDim, 0, stream>>>(
        A, B, C, M, K, N
    );
    
    CUDA_CHECK(cudaGetLastError());
}

/**
 * Alternative: Transposed B storage to avoid bank conflicts
 * 
 * Instead of padding, we can store B transposed in shared memory.
 * This changes the access pattern from column to row access.
 */
template<int TILE_SIZE>
__global__ void bank_conflict_free_transposed_sgemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // No padding needed if we transpose B
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float BsT[TILE_SIZE][TILE_SIZE + 1];  // B transposed
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        // Load A normally
        int aCol = t * TILE_SIZE + tx;
        if (row < M && aCol < K) {
            As[ty][tx] = A[row * K + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load B transposed: BsT[tx][ty] instead of Bs[ty][tx]
        int bRow = t * TILE_SIZE + ty;
        if (bRow < K && col < N) {
            BsT[tx][ty] = B[bRow * N + col];  // Note: indices swapped
        } else {
            BsT[tx][ty] = 0.0f;
        }
        
        __syncthreads();
        
        // Now both accesses are row-wise (no bank conflicts)
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * BsT[tx][k];  // Both row accesses
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

template<int TILE_SIZE = 32>
void launch_bank_conflict_free_transposed_sgemm(
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
    
    bank_conflict_free_transposed_sgemm_kernel<TILE_SIZE><<<gridDim, blockDim, 0, stream>>>(
        A, B, C, M, K, N
    );
    
    CUDA_CHECK(cudaGetLastError());
}
