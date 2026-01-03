#pragma once
/**
 * @file gemm.hpp
 * @brief GEMM (General Matrix Multiply) kernels with progressive optimization
 * 
 * Implements multiple GEMM versions from naive to tensor core optimized:
 * - v1: Naive implementation
 * - v2: Shared memory tiling
 * - v3: Double buffering
 * - v4: Tensor Core (WMMA) for CUDA 11.0+
 */

#include "../core/features.hpp"
#include "../core/cuda_check.hpp"
#include "../core/type_traits.hpp"
#include <algorithm>

#ifdef TC_HAS_WMMA
#include <mma.h>
#endif

namespace tensorcraft {
namespace kernels {

// ============================================================================
// GEMM Version Enumeration
// ============================================================================

enum class GemmVersion {
    Naive,          // Basic implementation
    Tiled,          // Shared memory tiling
    DoubleBuffer,   // Double buffering for latency hiding
    TensorCore,     // WMMA tensor core (CUDA 11.0+)
    Auto            // Automatic selection based on hardware
};

// ============================================================================
// GEMM v1: Naive Implementation
// ============================================================================

/**
 * @brief Naive GEMM kernel
 * 
 * C = alpha * A @ B + beta * C
 * A: [M, K], B: [K, N], C: [M, N]
 */
template<typename T>
__global__ void gemm_naive_kernel(
    const T* TC_RESTRICT A,
    const T* TC_RESTRICT B,
    T* TC_RESTRICT C,
    int M, int N, int K,
    T alpha, T beta) {
    
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        for (int k = 0; k < K; ++k) {
            sum += to_float(A[row * K + k]) * to_float(B[k * N + col]);
        }
        
        float c_val = to_float(C[row * N + col]);
        C[row * N + col] = from_float<T>(to_float(alpha) * sum + to_float(beta) * c_val);
    }
}

// ============================================================================
// GEMM v2: Shared Memory Tiling
// ============================================================================

/**
 * @brief Tiled GEMM kernel using shared memory
 * 
 * Uses tile-based approach to improve memory access patterns
 * and reduce global memory bandwidth requirements.
 */
template<typename T, int TILE_SIZE = 32>
__global__ void gemm_tiled_kernel(
    const T* TC_RESTRICT A,
    const T* TC_RESTRICT B,
    T* TC_RESTRICT C,
    int M, int N, int K,
    T alpha, T beta) {
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        // Load tiles into shared memory
        const int a_col = t * TILE_SIZE + tx;
        const int b_row = t * TILE_SIZE + ty;
        
        As[ty][tx] = (row < M && a_col < K) ? to_float(A[row * K + a_col]) : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? to_float(B[b_row * N + col]) : 0.0f;
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        float c_val = to_float(C[row * N + col]);
        C[row * N + col] = from_float<T>(to_float(alpha) * sum + to_float(beta) * c_val);
    }
}

// ============================================================================
// GEMM v3: Double Buffering
// ============================================================================

/**
 * @brief Double-buffered GEMM kernel
 * 
 * Uses double buffering to overlap memory loads with computation,
 * hiding memory latency for improved throughput.
 */
template<typename T, int TILE_SIZE = 32>
__global__ void gemm_double_buffer_kernel(
    const T* TC_RESTRICT A,
    const T* TC_RESTRICT B,
    T* TC_RESTRICT C,
    int M, int N, int K,
    T alpha, T beta) {
    
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx;
    
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    float sum = 0.0f;
    
    // Prefetch first tile
    int a_col = tx;
    int b_row = ty;
    As[0][ty][tx] = (row < M && a_col < K) ? to_float(A[row * K + a_col]) : 0.0f;
    Bs[0][ty][tx] = (b_row < K && col < N) ? to_float(B[b_row * N + col]) : 0.0f;
    __syncthreads();
    
    for (int t = 0; t < num_tiles; ++t) {
        const int curr = t % 2;
        const int next = (t + 1) % 2;
        
        // Prefetch next tile while computing current
        if (t + 1 < num_tiles) {
            a_col = (t + 1) * TILE_SIZE + tx;
            b_row = (t + 1) * TILE_SIZE + ty;
            As[next][ty][tx] = (row < M && a_col < K) ? to_float(A[row * K + a_col]) : 0.0f;
            Bs[next][ty][tx] = (b_row < K && col < N) ? to_float(B[b_row * N + col]) : 0.0f;
        }
        
        // Compute with current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[curr][ty][k] * Bs[curr][k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        float c_val = to_float(C[row * N + col]);
        C[row * N + col] = from_float<T>(to_float(alpha) * sum + to_float(beta) * c_val);
    }
}

// ============================================================================
// GEMM v4: Tensor Core (WMMA) - CUDA 11.0+
// ============================================================================

#ifdef TC_HAS_WMMA

/**
 * @brief WMMA-based GEMM kernel using Tensor Cores
 * 
 * Uses NVIDIA's WMMA API for hardware-accelerated matrix multiply.
 * Requires half precision input and supports FP32 accumulation.
 */
template<int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16>
__global__ void gemm_wmma_kernel(
    const half* TC_RESTRICT A,
    const half* TC_RESTRICT B,
    float* TC_RESTRICT C,
    int M, int N, int K,
    float alpha, float beta) {
    
    using namespace nvcuda::wmma;
    
    // Tile dimensions
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK_X = 4;
    constexpr int WARPS_PER_BLOCK_Y = 4;
    
    // Warp position
    const int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int warp_row = warp_id / WARPS_PER_BLOCK_X;
    const int warp_col = warp_id % WARPS_PER_BLOCK_X;
    
    // Global position
    const int block_row = blockIdx.y * WARPS_PER_BLOCK_Y * WMMA_M;
    const int block_col = blockIdx.x * WARPS_PER_BLOCK_X * WMMA_N;
    const int warp_row_offset = block_row + warp_row * WMMA_M;
    const int warp_col_offset = block_col + warp_col * WMMA_N;
    
    // Declare fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator
    fill_fragment(c_frag, 0.0f);
    
    // Accumulate over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        if (warp_row_offset < M && k < K) {
            load_matrix_sync(a_frag, A + warp_row_offset * K + k, K);
        }
        if (k < K && warp_col_offset < N) {
            load_matrix_sync(b_frag, B + k * N + warp_col_offset, N);
        }
        
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Scale by alpha
    for (int i = 0; i < c_frag.num_elements; ++i) {
        c_frag.x[i] *= alpha;
    }
    
    // Add beta * C if beta != 0
    if (beta != 0.0f) {
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_old_frag;
        if (warp_row_offset < M && warp_col_offset < N) {
            load_matrix_sync(c_old_frag, C + warp_row_offset * N + warp_col_offset, N, mem_row_major);
            for (int i = 0; i < c_frag.num_elements; ++i) {
                c_frag.x[i] += beta * c_old_frag.x[i];
            }
        }
    }
    
    // Store result
    if (warp_row_offset < M && warp_col_offset < N) {
        store_matrix_sync(C + warp_row_offset * N + warp_col_offset, c_frag, N, mem_row_major);
    }
}

#endif // TC_HAS_WMMA

// ============================================================================
// Matrix Transpose Kernels
// ============================================================================

/**
 * @brief Naive matrix transpose
 */
template<typename T>
__global__ void transpose_naive_kernel(
    const T* TC_RESTRICT input,
    T* TC_RESTRICT output,
    int rows, int cols) {
    
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

/**
 * @brief Optimized matrix transpose using shared memory
 * 
 * Uses shared memory with padding to avoid bank conflicts.
 */
template<typename T, int TILE_SIZE = 32>
__global__ void transpose_shared_kernel(
    const T* TC_RESTRICT input,
    T* TC_RESTRICT output,
    int rows, int cols) {
    
    // +1 padding to avoid bank conflicts
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    
    const int bx = blockIdx.x * TILE_SIZE;
    const int by = blockIdx.y * TILE_SIZE;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Load tile (coalesced read)
    const int in_row = by + ty;
    const int in_col = bx + tx;
    
    if (in_row < rows && in_col < cols) {
        tile[ty][tx] = to_float(input[in_row * cols + in_col]);
    }
    
    __syncthreads();
    
    // Write transposed tile (coalesced write)
    const int out_row = bx + ty;
    const int out_col = by + tx;
    
    if (out_row < cols && out_col < rows) {
        output[out_row * rows + out_col] = from_float<T>(tile[tx][ty]);
    }
}

// ============================================================================
// Launcher Functions
// ============================================================================

/**
 * @brief Launch GEMM kernel with version selection
 */
template<typename T>
void launch_gemm(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    T alpha = T(1), T beta = T(0),
    GemmVersion version = GemmVersion::Tiled,
    cudaStream_t stream = nullptr) {
    
    if (M == 0 || N == 0 || K == 0) return;
    
    constexpr int TILE_SIZE = 32;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    switch (version) {
        case GemmVersion::Naive:
            gemm_naive_kernel<T><<<grid, block, 0, stream>>>(
                A, B, C, M, N, K, alpha, beta);
            break;
            
        case GemmVersion::Tiled:
            gemm_tiled_kernel<T, TILE_SIZE><<<grid, block, 0, stream>>>(
                A, B, C, M, N, K, alpha, beta);
            break;
            
        case GemmVersion::DoubleBuffer:
            gemm_double_buffer_kernel<T, TILE_SIZE><<<grid, block, 0, stream>>>(
                A, B, C, M, N, K, alpha, beta);
            break;
            
        case GemmVersion::Auto:
        default:
            // Default to tiled version
            gemm_tiled_kernel<T, TILE_SIZE><<<grid, block, 0, stream>>>(
                A, B, C, M, N, K, alpha, beta);
            break;
    }
    
    TC_CUDA_CHECK_LAST();
}

#ifdef TC_HAS_WMMA
/**
 * @brief Launch WMMA GEMM kernel (half precision)
 */
inline void launch_gemm_wmma(
    const half* A, const half* B, float* C,
    int M, int N, int K,
    float alpha = 1.0f, float beta = 0.0f,
    cudaStream_t stream = nullptr) {
    
    if (M == 0 || N == 0 || K == 0) return;
    
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WARPS_X = 4;
    constexpr int WARPS_Y = 4;
    
    dim3 block(32 * WARPS_X, WARPS_Y);
    dim3 grid((N + WARPS_X * WMMA_N - 1) / (WARPS_X * WMMA_N),
              (M + WARPS_Y * WMMA_M - 1) / (WARPS_Y * WMMA_M));
    
    gemm_wmma_kernel<WMMA_M, WMMA_N, 16><<<grid, block, 0, stream>>>(
        A, B, C, M, N, K, alpha, beta);
    
    TC_CUDA_CHECK_LAST();
}
#endif

/**
 * @brief Launch matrix transpose
 */
template<typename T>
void launch_transpose(
    const T* input, T* output,
    int rows, int cols,
    bool use_shared = true,
    cudaStream_t stream = nullptr) {
    
    if (rows == 0 || cols == 0) return;
    
    constexpr int TILE_SIZE = 32;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    
    if (use_shared) {
        transpose_shared_kernel<T, TILE_SIZE><<<grid, block, 0, stream>>>(
            input, output, rows, cols);
    } else {
        transpose_naive_kernel<T><<<grid, block, 0, stream>>>(
            input, output, rows, cols);
    }
    
    TC_CUDA_CHECK_LAST();
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// GEMM with default parameters
template<typename T>
void gemm(
    const T* A, const T* B, T* C,
    size_t M, size_t N, size_t K,
    T alpha = T(1), T beta = T(0),
    cudaStream_t stream = nullptr) {
    
    launch_gemm(A, B, C, static_cast<int>(M), static_cast<int>(N), 
                static_cast<int>(K), alpha, beta, GemmVersion::Tiled, stream);
}

/// Matrix transpose
template<typename T>
void transpose(
    const T* input, T* output,
    size_t rows, size_t cols,
    cudaStream_t stream = nullptr) {
    
    launch_transpose(input, output, static_cast<int>(rows), 
                     static_cast<int>(cols), true, stream);
}

} // namespace kernels
} // namespace tensorcraft
