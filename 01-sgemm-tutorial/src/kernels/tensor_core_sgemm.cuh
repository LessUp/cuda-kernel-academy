#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "../utils/cuda_utils.cuh"

using namespace nvcuda;

/**
 * Tensor Core SGEMM Kernel using WMMA API
 * 
 * This implementation leverages NVIDIA Tensor Cores for maximum performance.
 * Tensor Cores are specialized hardware units that perform matrix multiply-
 * accumulate operations on small matrices (e.g., 16x16x16).
 * 
 * ============================================================================
 * Tensor Core Overview:
 * ============================================================================
 * 
 * - Available on Volta (sm_70) and newer architectures
 * - Perform D = A * B + C where:
 *   - A: 16x16 FP16
 *   - B: 16x16 FP16
 *   - C, D: 16x16 FP16 or FP32
 * - One warp (32 threads) cooperatively executes one WMMA operation
 * - Much higher throughput than CUDA cores for matrix operations
 * 
 * ============================================================================
 * WMMA API:
 * ============================================================================
 * 
 * 1. wmma::fragment - Holds matrix data distributed across warp threads
 * 2. wmma::load_matrix_sync - Load matrix from memory into fragment
 * 3. wmma::mma_sync - Perform matrix multiply-accumulate
 * 4. wmma::store_matrix_sync - Store fragment to memory
 * 
 * ============================================================================
 * Implementation Strategy:
 * ============================================================================
 * 
 * 1. Convert FP32 input to FP16 (Tensor Cores require FP16 input)
 * 2. Each warp computes one 16x16 output tile
 * 3. Iterate over K dimension in steps of 16
 * 4. Accumulate in FP32 for precision
 * 5. Store FP32 result
 * 
 * C = A * B
 * A: M x K (row-major, FP32 input converted to FP16)
 * B: K x N (row-major, FP32 input converted to FP16)
 * C: M x N (row-major, FP32 output)
 */

// WMMA tile dimensions
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

/**
 * Kernel to convert FP32 to FP16
 */
__global__ void float_to_half_kernel(
    const float* __restrict__ input,
    half* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

/**
 * Basic Tensor Core SGEMM Kernel
 * 
 * Each warp computes one WMMA_M x WMMA_N output tile.
 * Input matrices are expected to be in FP16 format.
 * 
 * Note: This kernel requires dimensions to be multiples of 16 for simplicity.
 * For non-aligned dimensions, use padding or a more sophisticated implementation.
 */
__global__ void tensor_core_sgemm_kernel_fp16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // Each block is one warp (32 threads)
    // blockIdx.x and blockIdx.y identify which 16x16 output tile this warp computes
    int warpM = blockIdx.y;
    int warpN = blockIdx.x;
    
    // Calculate starting positions for this warp's output tile
    int aRow = warpM * WMMA_M;
    int bCol = warpN * WMMA_N;
    
    // Bounds check - skip if this tile is completely outside the matrix
    if (aRow >= M || bCol >= N) return;
    
    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Iterate over K dimension in steps of WMMA_K (16)
    for (int k = 0; k < K; k += WMMA_K) {
        // Load A fragment: 16x16 tile of A starting at [aRow, k]
        // A is M x K, row-major, leading dimension is K
        wmma::load_matrix_sync(a_frag, A + aRow * K + k, K);
        
        // Load B fragment: 16x16 tile of B starting at [k, bCol]
        // B is K x N, row-major, leading dimension is N
        wmma::load_matrix_sync(b_frag, B + k * N + bCol, N);
        
        // Perform matrix multiply-accumulate: c_frag += a_frag * b_frag
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store result to C
    // C is M x N, row-major, leading dimension is N
    wmma::store_matrix_sync(C + aRow * N + bCol, c_frag, N, wmma::mem_row_major);
}

/**
 * Optimized Tensor Core SGEMM with shared memory staging
 */
template<int BLOCK_TILES_M = 4, int BLOCK_TILES_N = 4>
__global__ void tensor_core_sgemm_kernel_optimized(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // Shared memory for staging tiles
    constexpr int BLOCK_M = BLOCK_TILES_M * WMMA_M;
    constexpr int BLOCK_N = BLOCK_TILES_N * WMMA_N;
    constexpr int BLOCK_K = WMMA_K;
    
    __shared__ half As[BLOCK_M][BLOCK_K];
    __shared__ half Bs[BLOCK_K][BLOCK_N];
    
    // Warp identification within block
    int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int laneId = threadIdx.x % 32;
    
    int warpsPerBlockX = BLOCK_TILES_N;
    int warpX = warpId % warpsPerBlockX;
    int warpY = warpId / warpsPerBlockX;
    
    // Global position
    int blockRowStart = blockIdx.y * BLOCK_M;
    int blockColStart = blockIdx.x * BLOCK_N;
    
    // Fragments for this warp
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Iterate over K dimension
    for (int k = 0; k < K; k += BLOCK_K) {
        // Collaborative loading into shared memory
        // Each thread loads multiple elements
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int numThreads = blockDim.x * blockDim.y;
        
        // Load A tile
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += numThreads) {
            int row = i / BLOCK_K;
            int col = i % BLOCK_K;
            int globalRow = blockRowStart + row;
            int globalCol = k + col;
            
            if (globalRow < M && globalCol < K) {
                As[row][col] = A[globalRow * K + globalCol];
            } else {
                As[row][col] = __float2half(0.0f);
            }
        }
        
        // Load B tile
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += numThreads) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            int globalRow = k + row;
            int globalCol = blockColStart + col;
            
            if (globalRow < K && globalCol < N) {
                Bs[row][col] = B[globalRow * N + globalCol];
            } else {
                Bs[row][col] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
        // Load from shared memory and compute
        int warpRow = warpY * WMMA_M;
        int warpCol = warpX * WMMA_N;
        
        wmma::load_matrix_sync(a_frag, &As[warpRow][0], BLOCK_K);
        wmma::load_matrix_sync(b_frag, &Bs[0][warpCol], BLOCK_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        __syncthreads();
    }
    
    // Store result
    int cRow = blockRowStart + warpY * WMMA_M;
    int cCol = blockColStart + warpX * WMMA_N;
    
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

/**
 * Launch wrapper for Tensor Core SGEMM
 * Handles FP32 to FP16 conversion internally
 */
void launch_tensor_core_sgemm(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N,
    cudaStream_t stream = 0
) {
    // Allocate FP16 buffers
    half *d_A_fp16, *d_B_fp16;
    CUDA_CHECK(cudaMalloc(&d_A_fp16, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B_fp16, K * N * sizeof(half)));
    
    // Convert FP32 to FP16
    int blockSize = 256;
    int gridSizeA = (M * K + blockSize - 1) / blockSize;
    int gridSizeB = (K * N + blockSize - 1) / blockSize;
    
    float_to_half_kernel<<<gridSizeA, blockSize, 0, stream>>>(A, d_A_fp16, M * K);
    float_to_half_kernel<<<gridSizeB, blockSize, 0, stream>>>(B, d_B_fp16, K * N);
    
    // Launch Tensor Core kernel
    // Each warp handles one 16x16 tile
    // We need M/16 x N/16 warps total
    // Organize as blocks of warps
    
    // Simple configuration: one warp per 16x16 output tile
    // Block: 4x4 warps = 16 warps = 512 threads (but we use 2D indexing)
    dim3 blockDim(16, 16);  // 256 threads = 8 warps
    dim3 gridDim(
        (N + WMMA_N - 1) / WMMA_N,
        (M + WMMA_M - 1) / WMMA_M
    );
    
    // Adjust for warp-level indexing
    // Each "thread" in our grid is actually a warp position
    gridDim.x = (N + WMMA_N - 1) / WMMA_N;
    gridDim.y = (M + WMMA_M - 1) / WMMA_M;
    blockDim.x = 1;
    blockDim.y = 1;
    
    // Actually, let's use a simpler approach with proper warp handling
    // 4 warps per block (128 threads), each warp handles one 16x16 tile
    int warpsPerBlockX = 2;
    int warpsPerBlockY = 2;
    blockDim.x = warpsPerBlockX * 32;  // 64 threads
    blockDim.y = warpsPerBlockY;        // 2 rows of warps
    
    int tilesX = (N + WMMA_N - 1) / WMMA_N;
    int tilesY = (M + WMMA_M - 1) / WMMA_M;
    
    gridDim.x = (tilesX + warpsPerBlockX - 1) / warpsPerBlockX;
    gridDim.y = (tilesY + warpsPerBlockY - 1) / warpsPerBlockY;
    
    // Use the basic kernel for simplicity
    // Reset to simple 1 warp = 1 tile approach
    blockDim = dim3(32, 1);  // One warp
    gridDim = dim3(
        (N + WMMA_N - 1) / WMMA_N,
        (M + WMMA_M - 1) / WMMA_M
    );
    
    tensor_core_sgemm_kernel_fp16<<<gridDim, blockDim, 0, stream>>>(
        d_A_fp16, d_B_fp16, C, M, K, N
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    // Free FP16 buffers
    CUDA_CHECK(cudaFree(d_A_fp16));
    CUDA_CHECK(cudaFree(d_B_fp16));
}

/**
 * Tensor Core SGEMM with pre-converted FP16 inputs
 * Use this if you already have FP16 data
 */
void launch_tensor_core_sgemm_fp16(
    const half* A,
    const half* B,
    float* C,
    int M, int K, int N,
    cudaStream_t stream = 0
) {
    dim3 blockDim(32, 1);
    dim3 gridDim(
        (N + WMMA_N - 1) / WMMA_N,
        (M + WMMA_M - 1) / WMMA_M
    );
    
    tensor_core_sgemm_kernel_fp16<<<gridDim, blockDim, 0, stream>>>(
        A, B, C, M, K, N
    );
    
    CUDA_CHECK(cudaGetLastError());
}
