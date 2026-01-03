#pragma once

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

/**
 * Naive SGEMM Kernel
 * 
 * This is the simplest implementation of matrix multiplication on GPU.
 * Each thread computes one element of the output matrix C.
 * 
 * Performance Analysis:
 * - Each thread reads one row of A (K elements) and one column of B (K elements)
 * - Total global memory reads: 2 * M * N * K
 * - Total FLOPs: 2 * M * N * K
 * - Arithmetic Intensity: 1 FLOP/byte (very low, memory-bound)
 * 
 * Why it's slow:
 * 1. Non-coalesced memory access for matrix B (column access pattern)
 * 2. No data reuse - each element is read from global memory every time
 * 3. Low arithmetic intensity - severely memory bandwidth limited
 * 
 * C = A * B
 * A: M x K (row-major)
 * B: K x N (row-major)
 * C: M x N (row-major)
 */
__global__ void naive_sgemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Compute dot product of row of A and column of B
        for (int k = 0; k < K; ++k) {
            // A[row][k] * B[k][col]
            // A is accessed row-wise (coalesced within a warp for same row)
            // B is accessed column-wise (NOT coalesced - this is the main bottleneck)
            sum += A[row * K + k] * B[k * N + col];
        }
        
        // Write result to C
        C[row * N + col] = sum;
    }
}

/**
 * Launch wrapper for naive SGEMM kernel
 * 
 * @param A Device pointer to matrix A (M x K)
 * @param B Device pointer to matrix B (K x N)
 * @param C Device pointer to output matrix C (M x N)
 * @param M Number of rows in A and C
 * @param K Number of columns in A, rows in B
 * @param N Number of columns in B and C
 * @param stream CUDA stream (default: 0)
 */
template<int BLOCK_SIZE = 32>
void launch_naive_sgemm(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N,
    cudaStream_t stream = 0
) {
    // Configure grid and block dimensions
    // Each thread computes one element of C
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    
    // Launch kernel
    naive_sgemm_kernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, K, N);
    
    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());
}

/**
 * Naive SGEMM with alpha and beta scaling
 * C = alpha * A * B + beta * C
 */
__global__ void naive_sgemm_kernel_scaled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N,
    float alpha, float beta
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        // Apply scaling: C = alpha * A*B + beta * C
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

template<int BLOCK_SIZE = 32>
void launch_naive_sgemm_scaled(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N,
    float alpha, float beta,
    cudaStream_t stream = 0
) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    
    naive_sgemm_kernel_scaled<<<gridDim, blockDim, 0, stream>>>(
        A, B, C, M, K, N, alpha, beta
    );
    
    CUDA_CHECK(cudaGetLastError());
}
