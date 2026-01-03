#pragma once
/**
 * @file softmax.hpp
 * @brief Optimized Softmax kernel using online algorithm
 * 
 * Implements numerically stable softmax using the online algorithm
 * with warp shuffle reductions for optimal performance.
 */

#include "../core/features.hpp"
#include "../core/cuda_check.hpp"
#include "../core/type_traits.hpp"
#include <cfloat>

namespace tensorcraft {
namespace kernels {

// ============================================================================
// Warp Reduction Utilities
// ============================================================================

/**
 * @brief Warp-level max reduction using shuffle
 */
template<typename T>
TC_DEVICE_INLINE T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = val > other ? val : other;
    }
    return val;
}

/**
 * @brief Warp-level sum reduction using shuffle
 */
template<typename T>
TC_DEVICE_INLINE T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// Online Softmax Kernel
// ============================================================================

/**
 * @brief Online softmax kernel with warp shuffle reduction
 * 
 * Computes softmax in a single pass using the online algorithm:
 * 1. Find max while accumulating scaled exp sums
 * 2. Normalize in final pass
 * 
 * @tparam T Data type
 * @tparam BLOCK_SIZE Threads per block
 */
template<typename T, int BLOCK_SIZE = 256>
__global__ void softmax_kernel(
    const T* TC_RESTRICT input,
    T* TC_RESTRICT output,
    int rows,
    int cols) {
    
    const int row = blockIdx.x;
    if (row >= rows) return;
    
    const T* row_input = input + row * cols;
    T* row_output = output + row * cols;
    
    // Shared memory for cross-warp reduction
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];
    
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = BLOCK_SIZE / 32;
    
    // ========================================================================
    // Phase 1: Find row maximum
    // ========================================================================
    float thread_max = -FLT_MAX;
    
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        float val = to_float(row_input[i]);
        thread_max = fmaxf(thread_max, val);
    }
    
    // Warp reduce max
    thread_max = warp_reduce_max(thread_max);
    
    // Store warp results to shared memory
    if (lane == 0) {
        shared_max[warp_id] = thread_max;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        thread_max = (lane < num_warps) ? shared_max[lane] : -FLT_MAX;
        thread_max = warp_reduce_max(thread_max);
    }
    
    __shared__ float row_max;
    if (threadIdx.x == 0) {
        row_max = thread_max;
    }
    __syncthreads();
    
    // ========================================================================
    // Phase 2: Compute exp(x - max) and sum
    // ========================================================================
    float thread_sum = 0.0f;
    
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        float val = to_float(row_input[i]);
        float exp_val = expf(val - row_max);
        row_output[i] = from_float<T>(exp_val);  // Store intermediate
        thread_sum += exp_val;
    }
    
    // Warp reduce sum
    thread_sum = warp_reduce_sum(thread_sum);
    
    if (lane == 0) {
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        thread_sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        thread_sum = warp_reduce_sum(thread_sum);
    }
    
    __shared__ float row_sum;
    if (threadIdx.x == 0) {
        row_sum = thread_sum;
    }
    __syncthreads();
    
    // ========================================================================
    // Phase 3: Normalize
    // ========================================================================
    float inv_sum = 1.0f / row_sum;
    
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        float val = to_float(row_output[i]);
        row_output[i] = from_float<T>(val * inv_sum);
    }
}

/**
 * @brief Fused online softmax (single-pass for small sequences)
 * 
 * Uses online algorithm to compute softmax in a single pass,
 * maintaining running max and sum.
 */
template<typename T, int BLOCK_SIZE = 256>
__global__ void softmax_online_kernel(
    const T* TC_RESTRICT input,
    T* TC_RESTRICT output,
    int rows,
    int cols) {
    
    const int row = blockIdx.x;
    if (row >= rows) return;
    
    const T* row_input = input + row * cols;
    T* row_output = output + row * cols;
    
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];
    
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = BLOCK_SIZE / 32;
    
    // Online algorithm: maintain running max and sum
    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;
    
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        float val = to_float(row_input[i]);
        
        if (val > thread_max) {
            // Rescale previous sum
            thread_sum *= expf(thread_max - val);
            thread_max = val;
        }
        thread_sum += expf(val - thread_max);
    }
    
    // Combine across threads using online merge
    // First within warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, thread_sum, offset);
        
        if (other_max > thread_max) {
            thread_sum = thread_sum * expf(thread_max - other_max) + other_sum;
            thread_max = other_max;
        } else {
            thread_sum = thread_sum + other_sum * expf(other_max - thread_max);
        }
    }
    
    if (lane == 0) {
        shared_max[warp_id] = thread_max;
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        thread_max = (lane < num_warps) ? shared_max[lane] : -FLT_MAX;
        thread_sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
            float other_sum = __shfl_down_sync(0xffffffff, thread_sum, offset);
            
            if (other_max > thread_max) {
                thread_sum = thread_sum * expf(thread_max - other_max) + other_sum;
                thread_max = other_max;
            } else {
                thread_sum = thread_sum + other_sum * expf(other_max - thread_max);
            }
        }
    }
    
    __shared__ float row_max, row_sum;
    if (threadIdx.x == 0) {
        row_max = thread_max;
        row_sum = thread_sum;
    }
    __syncthreads();
    
    // Compute final softmax values
    float inv_sum = 1.0f / row_sum;
    
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        float val = to_float(row_input[i]);
        float result = expf(val - row_max) * inv_sum;
        row_output[i] = from_float<T>(result);
    }
}

// ============================================================================
// Launcher Functions
// ============================================================================

/**
 * @brief Launch softmax kernel
 * 
 * @param input Input tensor [rows, cols]
 * @param output Output tensor [rows, cols]
 * @param rows Number of rows
 * @param cols Number of columns (softmax dimension)
 * @param stream CUDA stream
 */
template<typename T>
void launch_softmax(
    const T* input,
    T* output,
    int rows,
    int cols,
    cudaStream_t stream = nullptr) {
    
    if (rows == 0 || cols == 0) return;
    
    constexpr int BLOCK_SIZE = 256;
    
    // Use online kernel for better numerical stability
    softmax_online_kernel<T, BLOCK_SIZE><<<rows, BLOCK_SIZE, 0, stream>>>(
        input, output, rows, cols);
    
    TC_CUDA_CHECK_LAST();
}

/**
 * @brief Softmax along last dimension for batched input
 * 
 * @param input Input tensor [batch, ..., dim]
 * @param output Output tensor [batch, ..., dim]
 * @param batch_size Total number of rows (product of all dims except last)
 * @param dim Softmax dimension size
 * @param stream CUDA stream
 */
template<typename T>
void softmax(
    const T* input,
    T* output,
    size_t batch_size,
    size_t dim,
    cudaStream_t stream = nullptr) {
    
    launch_softmax(input, output, static_cast<int>(batch_size), 
                   static_cast<int>(dim), stream);
}

} // namespace kernels
} // namespace tensorcraft
