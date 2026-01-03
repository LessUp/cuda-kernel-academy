#pragma once
/**
 * @file normalization.hpp
 * @brief Optimized normalization kernels (LayerNorm, RMSNorm, BatchNorm)
 * 
 * Implements efficient normalization layers using warp-level reductions
 * and vectorized memory access.
 */

#include "../core/features.hpp"
#include "../core/cuda_check.hpp"
#include "../core/type_traits.hpp"
#include "../memory/aligned_vector.hpp"
#include "softmax.hpp"  // For warp reduction utilities

namespace tensorcraft {
namespace kernels {

// ============================================================================
// LayerNorm Kernel
// ============================================================================

/**
 * @brief LayerNorm kernel with warp shuffle reduction
 * 
 * Computes: y = gamma * (x - mean) / sqrt(var + eps) + beta
 * 
 * @tparam T Data type
 * @tparam BLOCK_SIZE Threads per block
 */
template<typename T, int BLOCK_SIZE = 256>
__global__ void layernorm_kernel(
    const T* TC_RESTRICT input,
    const T* TC_RESTRICT gamma,
    const T* TC_RESTRICT beta,
    T* TC_RESTRICT output,
    int hidden_size,
    float eps) {
    
    const int row = blockIdx.x;
    const T* row_input = input + row * hidden_size;
    T* row_output = output + row * hidden_size;
    
    __shared__ float shared[32];
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = BLOCK_SIZE / 32;
    
    // ========================================================================
    // Phase 1: Compute mean
    // ========================================================================
    float thread_sum = 0.0f;
    
    for (int i = threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
        thread_sum += to_float(row_input[i]);
    }
    
    // Warp reduce
    thread_sum = warp_reduce_sum(thread_sum);
    
    if (lane == 0) {
        shared[warp_id] = thread_sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        thread_sum = (lane < num_warps) ? shared[lane] : 0.0f;
        thread_sum = warp_reduce_sum(thread_sum);
    }
    
    __shared__ float mean;
    if (threadIdx.x == 0) {
        mean = thread_sum / hidden_size;
    }
    __syncthreads();
    
    // ========================================================================
    // Phase 2: Compute variance
    // ========================================================================
    float thread_var = 0.0f;
    
    for (int i = threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
        float diff = to_float(row_input[i]) - mean;
        thread_var += diff * diff;
    }
    
    // Warp reduce
    thread_var = warp_reduce_sum(thread_var);
    
    if (lane == 0) {
        shared[warp_id] = thread_var;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        thread_var = (lane < num_warps) ? shared[lane] : 0.0f;
        thread_var = warp_reduce_sum(thread_var);
    }
    
    __shared__ float inv_std;
    if (threadIdx.x == 0) {
        inv_std = rsqrtf(thread_var / hidden_size + eps);
    }
    __syncthreads();
    
    // ========================================================================
    // Phase 3: Normalize and scale
    // ========================================================================
    for (int i = threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
        float val = to_float(row_input[i]);
        float normalized = (val - mean) * inv_std;
        float g = gamma ? to_float(gamma[i]) : 1.0f;
        float b = beta ? to_float(beta[i]) : 0.0f;
        row_output[i] = from_float<T>(normalized * g + b);
    }
}

/**
 * @brief LayerNorm without affine parameters
 */
template<typename T, int BLOCK_SIZE = 256>
__global__ void layernorm_no_affine_kernel(
    const T* TC_RESTRICT input,
    T* TC_RESTRICT output,
    int hidden_size,
    float eps) {
    
    const int row = blockIdx.x;
    const T* row_input = input + row * hidden_size;
    T* row_output = output + row * hidden_size;
    
    __shared__ float shared[32];
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = BLOCK_SIZE / 32;
    
    // Compute mean
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
        thread_sum += to_float(row_input[i]);
    }
    
    thread_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) shared[warp_id] = thread_sum;
    __syncthreads();
    
    if (warp_id == 0) {
        thread_sum = (lane < num_warps) ? shared[lane] : 0.0f;
        thread_sum = warp_reduce_sum(thread_sum);
    }
    
    __shared__ float mean;
    if (threadIdx.x == 0) mean = thread_sum / hidden_size;
    __syncthreads();
    
    // Compute variance
    float thread_var = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
        float diff = to_float(row_input[i]) - mean;
        thread_var += diff * diff;
    }
    
    thread_var = warp_reduce_sum(thread_var);
    if (lane == 0) shared[warp_id] = thread_var;
    __syncthreads();
    
    if (warp_id == 0) {
        thread_var = (lane < num_warps) ? shared[lane] : 0.0f;
        thread_var = warp_reduce_sum(thread_var);
    }
    
    __shared__ float inv_std;
    if (threadIdx.x == 0) inv_std = rsqrtf(thread_var / hidden_size + eps);
    __syncthreads();
    
    // Normalize
    for (int i = threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
        float val = to_float(row_input[i]);
        row_output[i] = from_float<T>((val - mean) * inv_std);
    }
}

// ============================================================================
// RMSNorm Kernel
// ============================================================================

/**
 * @brief RMSNorm kernel
 * 
 * Computes: y = x / RMS(x) * weight
 * where RMS(x) = sqrt(mean(x^2) + eps)
 */
template<typename T, int BLOCK_SIZE = 256>
__global__ void rmsnorm_kernel(
    const T* TC_RESTRICT input,
    const T* TC_RESTRICT weight,
    T* TC_RESTRICT output,
    int hidden_size,
    float eps) {
    
    const int row = blockIdx.x;
    const T* row_input = input + row * hidden_size;
    T* row_output = output + row * hidden_size;
    
    __shared__ float shared[32];
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = BLOCK_SIZE / 32;
    
    // Compute sum of squares
    float thread_sum = 0.0f;
    
    for (int i = threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
        float val = to_float(row_input[i]);
        thread_sum += val * val;
    }
    
    // Warp reduce
    thread_sum = warp_reduce_sum(thread_sum);
    
    if (lane == 0) {
        shared[warp_id] = thread_sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        thread_sum = (lane < num_warps) ? shared[lane] : 0.0f;
        thread_sum = warp_reduce_sum(thread_sum);
    }
    
    __shared__ float rms_inv;
    if (threadIdx.x == 0) {
        rms_inv = rsqrtf(thread_sum / hidden_size + eps);
    }
    __syncthreads();
    
    // Normalize and scale
    for (int i = threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
        float val = to_float(row_input[i]);
        float w = weight ? to_float(weight[i]) : 1.0f;
        row_output[i] = from_float<T>(val * rms_inv * w);
    }
}

// ============================================================================
// BatchNorm Kernel
// ============================================================================

/**
 * @brief BatchNorm forward kernel (inference mode)
 * 
 * Computes: y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
 */
template<typename T>
__global__ void batchnorm_inference_kernel(
    const T* TC_RESTRICT input,      // [N, C, H, W]
    const T* TC_RESTRICT gamma,      // [C]
    const T* TC_RESTRICT beta,       // [C]
    const float* TC_RESTRICT running_mean,  // [C]
    const float* TC_RESTRICT running_var,   // [C]
    T* TC_RESTRICT output,           // [N, C, H, W]
    int N, int C, int HW,
    float eps) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * HW;
    
    if (idx >= total) return;
    
    // Compute channel index
    const int c = (idx / HW) % C;
    
    float val = to_float(input[idx]);
    float mean = running_mean[c];
    float var = running_var[c];
    float g = gamma ? to_float(gamma[c]) : 1.0f;
    float b = beta ? to_float(beta[c]) : 0.0f;
    
    float normalized = (val - mean) * rsqrtf(var + eps);
    output[idx] = from_float<T>(normalized * g + b);
}

/**
 * @brief Fused BatchNorm + ReLU kernel
 */
template<typename T>
__global__ void batchnorm_relu_kernel(
    const T* TC_RESTRICT input,
    const T* TC_RESTRICT gamma,
    const T* TC_RESTRICT beta,
    const float* TC_RESTRICT running_mean,
    const float* TC_RESTRICT running_var,
    T* TC_RESTRICT output,
    int N, int C, int HW,
    float eps) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * HW;
    
    if (idx >= total) return;
    
    const int c = (idx / HW) % C;
    
    float val = to_float(input[idx]);
    float mean = running_mean[c];
    float var = running_var[c];
    float g = gamma ? to_float(gamma[c]) : 1.0f;
    float b = beta ? to_float(beta[c]) : 0.0f;
    
    float normalized = (val - mean) * rsqrtf(var + eps);
    float result = normalized * g + b;
    
    // Fused ReLU
    result = result > 0.0f ? result : 0.0f;
    
    output[idx] = from_float<T>(result);
}

// ============================================================================
// Launcher Functions
// ============================================================================

/**
 * @brief Launch LayerNorm kernel
 */
template<typename T>
void launch_layernorm(
    const T* input,
    const T* gamma,
    const T* beta,
    T* output,
    int batch_size,
    int hidden_size,
    float eps = 1e-5f,
    cudaStream_t stream = nullptr) {
    
    if (batch_size == 0 || hidden_size == 0) return;
    
    constexpr int BLOCK_SIZE = 256;
    
    if (gamma && beta) {
        layernorm_kernel<T, BLOCK_SIZE><<<batch_size, BLOCK_SIZE, 0, stream>>>(
            input, gamma, beta, output, hidden_size, eps);
    } else {
        layernorm_no_affine_kernel<T, BLOCK_SIZE><<<batch_size, BLOCK_SIZE, 0, stream>>>(
            input, output, hidden_size, eps);
    }
    
    TC_CUDA_CHECK_LAST();
}

/**
 * @brief Launch RMSNorm kernel
 */
template<typename T>
void launch_rmsnorm(
    const T* input,
    const T* weight,
    T* output,
    int batch_size,
    int hidden_size,
    float eps = 1e-6f,
    cudaStream_t stream = nullptr) {
    
    if (batch_size == 0 || hidden_size == 0) return;
    
    constexpr int BLOCK_SIZE = 256;
    
    rmsnorm_kernel<T, BLOCK_SIZE><<<batch_size, BLOCK_SIZE, 0, stream>>>(
        input, weight, output, hidden_size, eps);
    
    TC_CUDA_CHECK_LAST();
}

/**
 * @brief Launch BatchNorm inference kernel
 */
template<typename T>
void launch_batchnorm(
    const T* input,
    const T* gamma,
    const T* beta,
    const float* running_mean,
    const float* running_var,
    T* output,
    int N, int C, int H, int W,
    float eps = 1e-5f,
    bool fuse_relu = false,
    cudaStream_t stream = nullptr) {
    
    const int total = N * C * H * W;
    if (total == 0) return;
    
    constexpr int BLOCK_SIZE = 256;
    const int grid_size = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    if (fuse_relu) {
        batchnorm_relu_kernel<T><<<grid_size, BLOCK_SIZE, 0, stream>>>(
            input, gamma, beta, running_mean, running_var, output,
            N, C, H * W, eps);
    } else {
        batchnorm_inference_kernel<T><<<grid_size, BLOCK_SIZE, 0, stream>>>(
            input, gamma, beta, running_mean, running_var, output,
            N, C, H * W, eps);
    }
    
    TC_CUDA_CHECK_LAST();
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// LayerNorm
template<typename T>
void layernorm(
    const T* input,
    const T* gamma,
    const T* beta,
    T* output,
    size_t batch_size,
    size_t hidden_size,
    float eps = 1e-5f,
    cudaStream_t stream = nullptr) {
    
    launch_layernorm(input, gamma, beta, output,
                     static_cast<int>(batch_size),
                     static_cast<int>(hidden_size),
                     eps, stream);
}

/// RMSNorm
template<typename T>
void rmsnorm(
    const T* input,
    const T* weight,
    T* output,
    size_t batch_size,
    size_t hidden_size,
    float eps = 1e-6f,
    cudaStream_t stream = nullptr) {
    
    launch_rmsnorm(input, weight, output,
                   static_cast<int>(batch_size),
                   static_cast<int>(hidden_size),
                   eps, stream);
}

} // namespace kernels
} // namespace tensorcraft
