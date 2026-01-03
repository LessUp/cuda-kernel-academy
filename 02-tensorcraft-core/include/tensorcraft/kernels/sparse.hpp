#pragma once
/**
 * @file sparse.hpp
 * @brief Sparse matrix operations (CSR/CSC formats, SpMV, SpMM)
 */

#include "../core/features.hpp"
#include "../core/cuda_check.hpp"
#include "../core/type_traits.hpp"

namespace tensorcraft {
namespace kernels {

// ============================================================================
// CSR (Compressed Sparse Row) Format
// ============================================================================

template<typename T>
struct CSRMatrix {
    T* values;           // Non-zero values [nnz]
    int* col_indices;    // Column indices [nnz]
    int* row_ptrs;       // Row pointers [rows + 1]
    int rows;
    int cols;
    int nnz;
};

// ============================================================================
// CSC (Compressed Sparse Column) Format
// ============================================================================

template<typename T>
struct CSCMatrix {
    T* values;           // Non-zero values [nnz]
    int* row_indices;    // Row indices [nnz]
    int* col_ptrs;       // Column pointers [cols + 1]
    int rows;
    int cols;
    int nnz;
};

// ============================================================================
// Dense to CSR Conversion
// ============================================================================

template<typename T>
__global__ void count_nnz_per_row_kernel(
    const T* TC_RESTRICT dense,
    int* TC_RESTRICT row_counts,
    int rows, int cols, T threshold) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    int count = 0;
    for (int col = 0; col < cols; ++col) {
        T val = dense[row * cols + col];
        if (fabsf(to_float(val)) > to_float(threshold)) {
            ++count;
        }
    }
    row_counts[row] = count;
}

template<typename T>
__global__ void dense_to_csr_kernel(
    const T* TC_RESTRICT dense,
    T* TC_RESTRICT values,
    int* TC_RESTRICT col_indices,
    const int* TC_RESTRICT row_ptrs,
    int rows, int cols, T threshold) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    int write_idx = row_ptrs[row];
    for (int col = 0; col < cols; ++col) {
        T val = dense[row * cols + col];
        if (fabsf(to_float(val)) > to_float(threshold)) {
            values[write_idx] = val;
            col_indices[write_idx] = col;
            ++write_idx;
        }
    }
}

// ============================================================================
// CSR to Dense Conversion
// ============================================================================

template<typename T>
__global__ void csr_to_dense_kernel(
    const T* TC_RESTRICT values,
    const int* TC_RESTRICT col_indices,
    const int* TC_RESTRICT row_ptrs,
    T* TC_RESTRICT dense,
    int rows, int cols) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    // Initialize row to zero
    for (int col = 0; col < cols; ++col) {
        dense[row * cols + col] = T(0);
    }
    
    // Fill non-zero values
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];
    for (int i = start; i < end; ++i) {
        int col = col_indices[i];
        dense[row * cols + col] = values[i];
    }
}

// ============================================================================
// SpMV (Sparse Matrix-Vector Multiplication)
// ============================================================================

/**
 * @brief CSR SpMV kernel: y = A * x
 */
template<typename T>
__global__ void spmv_csr_kernel(
    const T* TC_RESTRICT values,
    const int* TC_RESTRICT col_indices,
    const int* TC_RESTRICT row_ptrs,
    const T* TC_RESTRICT x,
    T* TC_RESTRICT y,
    int rows) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    float sum = 0.0f;
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];
    
    for (int i = start; i < end; ++i) {
        int col = col_indices[i];
        sum += to_float(values[i]) * to_float(x[col]);
    }
    
    y[row] = from_float<T>(sum);
}

/**
 * @brief Vectorized CSR SpMV using warp reduction
 */
template<typename T, int WARP_SIZE = 32>
__global__ void spmv_csr_vector_kernel(
    const T* TC_RESTRICT values,
    const int* TC_RESTRICT col_indices,
    const int* TC_RESTRICT row_ptrs,
    const T* TC_RESTRICT x,
    T* TC_RESTRICT y,
    int rows) {
    
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    
    if (warp_id >= rows) return;
    
    int start = row_ptrs[warp_id];
    int end = row_ptrs[warp_id + 1];
    
    float sum = 0.0f;
    for (int i = start + lane; i < end; i += WARP_SIZE) {
        int col = col_indices[i];
        sum += to_float(values[i]) * to_float(x[col]);
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane == 0) {
        y[warp_id] = from_float<T>(sum);
    }
}

// ============================================================================
// SpMM (Sparse Matrix-Matrix Multiplication)
// ============================================================================

/**
 * @brief CSR SpMM kernel: C = A * B
 * A: sparse [M, K], B: dense [K, N], C: dense [M, N]
 */
template<typename T>
__global__ void spmm_csr_kernel(
    const T* TC_RESTRICT A_values,
    const int* TC_RESTRICT A_col_indices,
    const int* TC_RESTRICT A_row_ptrs,
    const T* TC_RESTRICT B,
    T* TC_RESTRICT C,
    int M, int K, int N) {
    
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    int start = A_row_ptrs[row];
    int end = A_row_ptrs[row + 1];
    
    for (int i = start; i < end; ++i) {
        int k = A_col_indices[i];
        sum += to_float(A_values[i]) * to_float(B[k * N + col]);
    }
    
    C[row * N + col] = from_float<T>(sum);
}

/**
 * @brief Tiled CSR SpMM for better memory access
 */
template<typename T, int TILE_N = 32>
__global__ void spmm_csr_tiled_kernel(
    const T* TC_RESTRICT A_values,
    const int* TC_RESTRICT A_col_indices,
    const int* TC_RESTRICT A_row_ptrs,
    const T* TC_RESTRICT B,
    T* TC_RESTRICT C,
    int M, int K, int N) {
    
    __shared__ float B_shared[32][TILE_N + 1];  // +1 to avoid bank conflicts
    
    int row = blockIdx.y;
    int tile_col = blockIdx.x * TILE_N;
    int tx = threadIdx.x;
    
    if (row >= M) return;
    
    float sums[TILE_N / 32];
    #pragma unroll
    for (int i = 0; i < TILE_N / 32; ++i) {
        sums[i] = 0.0f;
    }
    
    int start = A_row_ptrs[row];
    int end = A_row_ptrs[row + 1];
    
    // Process in chunks of 32 K values
    for (int k_base = 0; k_base < K; k_base += 32) {
        // Load B tile into shared memory
        for (int n = 0; n < TILE_N && tile_col + n < N; ++n) {
            int k = k_base + tx;
            B_shared[tx][n] = (k < K) ? to_float(B[k * N + tile_col + n]) : 0.0f;
        }
        __syncthreads();
        
        // Accumulate
        for (int i = start; i < end; ++i) {
            int k = A_col_indices[i];
            if (k >= k_base && k < k_base + 32) {
                float a_val = to_float(A_values[i]);
                int k_local = k - k_base;
                for (int n = 0; n < TILE_N / 32; ++n) {
                    int col = tx + n * 32;
                    if (tile_col + col < N) {
                        sums[n] += a_val * B_shared[k_local][col];
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Write results
    for (int n = 0; n < TILE_N / 32; ++n) {
        int col = tile_col + tx + n * 32;
        if (col < N) {
            C[row * N + col] = from_float<T>(sums[n]);
        }
    }
}

// ============================================================================
// Launcher Functions
// ============================================================================

template<typename T>
void launch_spmv_csr(
    const T* values, const int* col_indices, const int* row_ptrs,
    const T* x, T* y, int rows,
    bool use_vector = true,
    cudaStream_t stream = nullptr) {
    
    if (rows == 0) return;
    
    if (use_vector) {
        int warps_per_block = 4;
        int threads = warps_per_block * 32;
        int blocks = (rows + warps_per_block - 1) / warps_per_block;
        spmv_csr_vector_kernel<T><<<blocks, threads, 0, stream>>>(
            values, col_indices, row_ptrs, x, y, rows);
    } else {
        int block_size = 256;
        int grid_size = (rows + block_size - 1) / block_size;
        spmv_csr_kernel<T><<<grid_size, block_size, 0, stream>>>(
            values, col_indices, row_ptrs, x, y, rows);
    }
    TC_CUDA_CHECK_LAST();
}

template<typename T>
void launch_spmm_csr(
    const T* A_values, const int* A_col_indices, const int* A_row_ptrs,
    const T* B, T* C,
    int M, int K, int N,
    cudaStream_t stream = nullptr) {
    
    if (M == 0 || N == 0) return;
    
    constexpr int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, M);
    
    spmm_csr_kernel<T><<<grid, block, 0, stream>>>(
        A_values, A_col_indices, A_row_ptrs, B, C, M, K, N);
    TC_CUDA_CHECK_LAST();
}

template<typename T>
void launch_csr_to_dense(
    const T* values, const int* col_indices, const int* row_ptrs,
    T* dense, int rows, int cols,
    cudaStream_t stream = nullptr) {
    
    if (rows == 0) return;
    
    int block_size = 256;
    int grid_size = (rows + block_size - 1) / block_size;
    csr_to_dense_kernel<T><<<grid_size, block_size, 0, stream>>>(
        values, col_indices, row_ptrs, dense, rows, cols);
    TC_CUDA_CHECK_LAST();
}

// Convenience functions
template<typename T>
void spmv(const CSRMatrix<T>& A, const T* x, T* y, cudaStream_t stream = nullptr) {
    launch_spmv_csr(A.values, A.col_indices, A.row_ptrs, x, y, A.rows, true, stream);
}

template<typename T>
void spmm(const CSRMatrix<T>& A, const T* B, T* C, int N, cudaStream_t stream = nullptr) {
    launch_spmm_csr(A.values, A.col_indices, A.row_ptrs, B, C, A.rows, A.cols, N, stream);
}

} // namespace kernels
} // namespace tensorcraft
