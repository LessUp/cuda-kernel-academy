#pragma once
/**
 * @file fusion.hpp
 * @brief Fused operators and quantization support
 * 
 * Implements:
 * - Bias + Activation fusion (epilogue functors)
 * - INT8/FP8 quantization
 */

#include "../core/features.hpp"
#include "../core/cuda_check.hpp"
#include "../core/type_traits.hpp"
#include "elementwise.hpp"

namespace tensorcraft {
namespace kernels {

// ============================================================================
// Epilogue Functors for GEMM Fusion
// ============================================================================

/// Identity epilogue (no-op)
struct EpilogueIdentity {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T val, int row, int col) const {
        return val;
    }
};

/// Bias addition epilogue
template<typename T>
struct EpilogueBias {
    const T* bias;
    
    TC_HOST_DEVICE EpilogueBias(const T* b) : bias(b) {}
    
    TC_DEVICE_INLINE T operator()(T val, int row, int col) const {
        return val + bias[col];
    }
};

/// Bias + ReLU epilogue
template<typename T>
struct EpilogueBiasReLU {
    const T* bias;
    
    TC_HOST_DEVICE EpilogueBiasReLU(const T* b) : bias(b) {}
    
    TC_DEVICE_INLINE T operator()(T val, int row, int col) const {
        T result = val + bias[col];
        return result > T(0) ? result : T(0);
    }
};

/// Bias + GeLU epilogue
template<typename T>
struct EpilogueBiasGeLU {
    const T* bias;
    
    TC_HOST_DEVICE EpilogueBiasGeLU(const T* b) : bias(b) {}
    
    TC_DEVICE_INLINE T operator()(T val, int row, int col) const {
        float x = to_float(val) + to_float(bias[col]);
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;
        constexpr float coeff = 0.044715f;
        float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        float result = 0.5f * x * (1.0f + tanhf(inner));
        return from_float<T>(result);
    }
};

/// Bias + SiLU epilogue
template<typename T>
struct EpilogueBiasSiLU {
    const T* bias;
    
    TC_HOST_DEVICE EpilogueBiasSiLU(const T* b) : bias(b) {}
    
    TC_DEVICE_INLINE T operator()(T val, int row, int col) const {
        float x = to_float(val) + to_float(bias[col]);
        float result = x / (1.0f + expf(-x));
        return from_float<T>(result);
    }
};

// ============================================================================
// Fused GEMM + Epilogue Kernel
// ============================================================================

template<typename T, typename Epilogue, int TILE_SIZE = 32>
__global__ void gemm_fused_kernel(
    const T* TC_RESTRICT A,
    const T* TC_RESTRICT B,
    T* TC_RESTRICT C,
    int M, int N, int K,
    T alpha,
    Epilogue epilogue) {
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;
        
        As[ty][tx] = (row < M && a_col < K) ? to_float(A[row * K + a_col]) : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? to_float(B[b_row * N + col]) : 0.0f;
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        T result = from_float<T>(to_float(alpha) * sum);
        C[row * N + col] = epilogue(result, row, col);
    }
}

// ============================================================================
// INT8 Quantization
// ============================================================================

/// Quantization parameters
struct QuantParams {
    float scale;
    int zero_point;
};

/// Quantize FP32 to INT8
template<typename T>
__global__ void quantize_int8_kernel(
    const T* TC_RESTRICT input,
    int8_t* TC_RESTRICT output,
    float scale, int zero_point,
    size_t n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float val = to_float(input[idx]);
    int quantized = __float2int_rn(val / scale) + zero_point;
    quantized = max(-128, min(127, quantized));
    output[idx] = static_cast<int8_t>(quantized);
}

/// Dequantize INT8 to FP32
template<typename T>
__global__ void dequantize_int8_kernel(
    const int8_t* TC_RESTRICT input,
    T* TC_RESTRICT output,
    float scale, int zero_point,
    size_t n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float val = (static_cast<float>(input[idx]) - zero_point) * scale;
    output[idx] = from_float<T>(val);
}

/// Compute quantization parameters (min-max)
template<typename T>
__global__ void compute_quant_params_kernel(
    const T* TC_RESTRICT input,
    float* TC_RESTRICT min_val,
    float* TC_RESTRICT max_val,
    size_t n) {
    
    __shared__ float s_min[256];
    __shared__ float s_max[256];
    
    int tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        float val = to_float(input[i]);
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }
    
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();
    
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMin(reinterpret_cast<int*>(min_val), __float_as_int(s_min[0]));
        atomicMax(reinterpret_cast<int*>(max_val), __float_as_int(s_max[0]));
    }
}

// ============================================================================
// FP8 Quantization (CUDA 12.0+)
// ============================================================================

#ifdef TC_HAS_FP8
#include <cuda_fp8.h>

/// Quantize FP32 to FP8 E4M3
template<typename T>
__global__ void quantize_fp8_e4m3_kernel(
    const T* TC_RESTRICT input,
    __nv_fp8_e4m3* TC_RESTRICT output,
    float scale,
    size_t n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float val = to_float(input[idx]) / scale;
    output[idx] = __nv_fp8_e4m3(val);
}

/// Dequantize FP8 E4M3 to FP32
template<typename T>
__global__ void dequantize_fp8_e4m3_kernel(
    const __nv_fp8_e4m3* TC_RESTRICT input,
    T* TC_RESTRICT output,
    float scale,
    size_t n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float val = float(input[idx]) * scale;
    output[idx] = from_float<T>(val);
}

#endif // TC_HAS_FP8

// ============================================================================
// Launcher Functions
// ============================================================================

template<typename T, typename Epilogue>
void launch_gemm_fused(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    T alpha,
    Epilogue epilogue,
    cudaStream_t stream = nullptr) {
    
    if (M == 0 || N == 0 || K == 0) return;
    
    constexpr int TILE_SIZE = 32;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    gemm_fused_kernel<T, Epilogue, TILE_SIZE><<<grid, block, 0, stream>>>(
        A, B, C, M, N, K, alpha, epilogue);
    TC_CUDA_CHECK_LAST();
}

template<typename T>
void launch_quantize_int8(
    const T* input, int8_t* output,
    float scale, int zero_point,
    size_t n,
    cudaStream_t stream = nullptr) {
    
    if (n == 0) return;
    
    constexpr int BLOCK_SIZE = 256;
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    quantize_int8_kernel<T><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        input, output, scale, zero_point, n);
    TC_CUDA_CHECK_LAST();
}

template<typename T>
void launch_dequantize_int8(
    const int8_t* input, T* output,
    float scale, int zero_point,
    size_t n,
    cudaStream_t stream = nullptr) {
    
    if (n == 0) return;
    
    constexpr int BLOCK_SIZE = 256;
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    dequantize_int8_kernel<T><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        input, output, scale, zero_point, n);
    TC_CUDA_CHECK_LAST();
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// GEMM with Bias + GeLU fusion
template<typename T>
void gemm_bias_gelu(
    const T* A, const T* B, const T* bias, T* C,
    size_t M, size_t N, size_t K,
    cudaStream_t stream = nullptr) {
    
    EpilogueBiasGeLU<T> epilogue(bias);
    launch_gemm_fused(A, B, C, (int)M, (int)N, (int)K, T(1), epilogue, stream);
}

/// GEMM with Bias + ReLU fusion
template<typename T>
void gemm_bias_relu(
    const T* A, const T* B, const T* bias, T* C,
    size_t M, size_t N, size_t K,
    cudaStream_t stream = nullptr) {
    
    EpilogueBiasReLU<T> epilogue(bias);
    launch_gemm_fused(A, B, C, (int)M, (int)N, (int)K, T(1), epilogue, stream);
}

/// Quantize to INT8
template<typename T>
void quantize_int8(
    const T* input, int8_t* output,
    float scale, int zero_point, size_t n,
    cudaStream_t stream = nullptr) {
    
    launch_quantize_int8(input, output, scale, zero_point, n, stream);
}

/// Dequantize from INT8
template<typename T>
void dequantize_int8(
    const int8_t* input, T* output,
    float scale, int zero_point, size_t n,
    cudaStream_t stream = nullptr) {
    
    launch_dequantize_int8(input, output, scale, zero_point, n, stream);
}

} // namespace kernels
} // namespace tensorcraft
