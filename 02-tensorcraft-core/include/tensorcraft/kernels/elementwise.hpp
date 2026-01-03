#pragma once
/**
 * @file elementwise.hpp
 * @brief Generic elementwise kernel framework with vectorized access
 * 
 * Provides a flexible framework for implementing elementwise operations
 * using functor pattern and vectorized memory access for optimal bandwidth.
 */

#include "../core/features.hpp"
#include "../core/type_traits.hpp"
#include "../core/cuda_check.hpp"
#include "../memory/aligned_vector.hpp"
#include <algorithm>
#include <cmath>

namespace tensorcraft {
namespace kernels {

// ============================================================================
// Generic Elementwise Kernel
// ============================================================================

/**
 * @brief Vectorized elementwise kernel with functor
 * 
 * @tparam T Data type
 * @tparam Func Functor type with operator()(T) -> T
 * @tparam VecSize Vector size for memory access
 */
template<typename T, typename Func, int VecSize = 4>
__global__ void elementwise_kernel(
    const T* TC_RESTRICT input,
    T* TC_RESTRICT output,
    size_t n,
    Func func) {
    
    using VecT = AlignedVector<T, VecSize>;
    
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
    const size_t stride = blockDim.x * gridDim.x * VecSize;
    
    // Vectorized processing
    for (size_t i = idx; i + VecSize <= n; i += stride) {
        VecT in_vec = *reinterpret_cast<const VecT*>(&input[i]);
        VecT out_vec;
        
        #pragma unroll
        for (int k = 0; k < VecSize; ++k) {
            out_vec[k] = func(in_vec[k]);
        }
        
        *reinterpret_cast<VecT*>(&output[i]) = out_vec;
    }
    
    // Handle tail elements
    const size_t tail_start = (n / VecSize) * VecSize;
    for (size_t i = tail_start + threadIdx.x; i < n; i += blockDim.x) {
        if (blockIdx.x == 0) {
            output[i] = func(input[i]);
        }
    }
}

/**
 * @brief Naive (non-vectorized) elementwise kernel
 */
template<typename T, typename Func>
__global__ void elementwise_kernel_naive(
    const T* TC_RESTRICT input,
    T* TC_RESTRICT output,
    size_t n,
    Func func) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        output[i] = func(input[i]);
    }
}

/**
 * @brief Binary elementwise kernel (two inputs)
 */
template<typename T, typename Func, int VecSize = 4>
__global__ void elementwise_binary_kernel(
    const T* TC_RESTRICT input1,
    const T* TC_RESTRICT input2,
    T* TC_RESTRICT output,
    size_t n,
    Func func) {
    
    using VecT = AlignedVector<T, VecSize>;
    
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
    const size_t stride = blockDim.x * gridDim.x * VecSize;
    
    for (size_t i = idx; i + VecSize <= n; i += stride) {
        VecT in1_vec = *reinterpret_cast<const VecT*>(&input1[i]);
        VecT in2_vec = *reinterpret_cast<const VecT*>(&input2[i]);
        VecT out_vec;
        
        #pragma unroll
        for (int k = 0; k < VecSize; ++k) {
            out_vec[k] = func(in1_vec[k], in2_vec[k]);
        }
        
        *reinterpret_cast<VecT*>(&output[i]) = out_vec;
    }
    
    // Handle tail
    const size_t tail_start = (n / VecSize) * VecSize;
    for (size_t i = tail_start + threadIdx.x; i < n; i += blockDim.x) {
        if (blockIdx.x == 0) {
            output[i] = func(input1[i], input2[i]);
        }
    }
}

// ============================================================================
// Activation Function Functors
// ============================================================================

/**
 * @brief ReLU activation: max(0, x)
 */
struct ReLU {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T x) const {
        return x > T(0) ? x : T(0);
    }
};

/**
 * @brief SiLU (Swish) activation: x * sigmoid(x)
 */
struct SiLU {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T x) const {
        float xf = to_float(x);
        float result = xf / (1.0f + expf(-xf));
        return from_float<T>(result);
    }
};

/**
 * @brief GeLU activation (approximate)
 * GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
struct GeLU {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T x) const {
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;
        constexpr float coeff = 0.044715f;
        
        float xf = to_float(x);
        float inner = sqrt_2_over_pi * (xf + coeff * xf * xf * xf);
        float result = 0.5f * xf * (1.0f + tanhf(inner));
        return from_float<T>(result);
    }
};

/**
 * @brief GeLU activation (exact using erf)
 * GeLU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
 */
struct GeLUExact {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T x) const {
        constexpr float inv_sqrt2 = 0.7071067811865476f;
        float xf = to_float(x);
        float result = 0.5f * xf * (1.0f + erff(xf * inv_sqrt2));
        return from_float<T>(result);
    }
};

/**
 * @brief Leaky ReLU: x if x > 0, else alpha * x
 */
template<typename T = float>
struct LeakyReLU {
    T alpha;
    
    TC_HOST_DEVICE LeakyReLU(T alpha = T(0.01)) : alpha(alpha) {}
    
    TC_DEVICE_INLINE T operator()(T x) const {
        return x > T(0) ? x : alpha * x;
    }
};

/**
 * @brief ELU activation: x if x > 0, else alpha * (exp(x) - 1)
 */
template<typename T = float>
struct ELU {
    T alpha;
    
    TC_HOST_DEVICE ELU(T alpha = T(1.0)) : alpha(alpha) {}
    
    TC_DEVICE_INLINE T operator()(T x) const {
        float xf = to_float(x);
        float af = to_float(alpha);
        float result = xf > 0.0f ? xf : af * (expf(xf) - 1.0f);
        return from_float<T>(result);
    }
};

/**
 * @brief Swish activation: x * sigmoid(beta * x)
 */
template<typename T = float>
struct Swish {
    T beta;
    
    TC_HOST_DEVICE Swish(T beta = T(1.0)) : beta(beta) {}
    
    TC_DEVICE_INLINE T operator()(T x) const {
        float xf = to_float(x);
        float bf = to_float(beta);
        float result = xf / (1.0f + expf(-bf * xf));
        return from_float<T>(result);
    }
};

/**
 * @brief Sigmoid activation: 1 / (1 + exp(-x))
 */
struct Sigmoid {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T x) const {
        float xf = to_float(x);
        float result = 1.0f / (1.0f + expf(-xf));
        return from_float<T>(result);
    }
};

/**
 * @brief Tanh activation
 */
struct Tanh {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T x) const {
        float xf = to_float(x);
        return from_float<T>(tanhf(xf));
    }
};

/**
 * @brief Softplus activation: log(1 + exp(x))
 */
struct Softplus {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T x) const {
        float xf = to_float(x);
        // Numerically stable version
        float result = xf > 20.0f ? xf : logf(1.0f + expf(xf));
        return from_float<T>(result);
    }
};

// ============================================================================
// Binary Operation Functors
// ============================================================================

/**
 * @brief Addition: a + b
 */
struct Add {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T a, T b) const {
        return a + b;
    }
};

/**
 * @brief Subtraction: a - b
 */
struct Sub {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T a, T b) const {
        return a - b;
    }
};

/**
 * @brief Multiplication: a * b
 */
struct Mul {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T a, T b) const {
        return a * b;
    }
};

/**
 * @brief Division: a / b
 */
struct Div {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T a, T b) const {
        return a / b;
    }
};

/**
 * @brief Maximum: max(a, b)
 */
struct Max {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T a, T b) const {
        return a > b ? a : b;
    }
};

/**
 * @brief Minimum: min(a, b)
 */
struct Min {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T a, T b) const {
        return a < b ? a : b;
    }
};

// ============================================================================
// Launcher Functions
// ============================================================================

/**
 * @brief Launch elementwise kernel with automatic configuration
 */
template<typename T, typename Func>
void launch_elementwise(
    const T* input,
    T* output,
    size_t n,
    Func func,
    cudaStream_t stream = nullptr,
    bool use_vectorized = true) {
    
    if (n == 0) return;
    
    constexpr int block_size = 256;
    constexpr int vec_size = optimal_vec_size<T>();
    
    if (use_vectorized && is_aligned<T, vec_size>(input) && is_aligned<T, vec_size>(output)) {
        int grid_size = static_cast<int>((n + block_size * vec_size - 1) / (block_size * vec_size));
        grid_size = std::min(grid_size, 65535);
        
        elementwise_kernel<T, Func, vec_size><<<grid_size, block_size, 0, stream>>>(
            input, output, n, func);
    } else {
        int grid_size = static_cast<int>((n + block_size - 1) / block_size);
        grid_size = std::min(grid_size, 65535);
        
        elementwise_kernel_naive<T, Func><<<grid_size, block_size, 0, stream>>>(
            input, output, n, func);
    }
    
    TC_CUDA_CHECK_LAST();
}

/**
 * @brief Launch binary elementwise kernel
 */
template<typename T, typename Func>
void launch_elementwise_binary(
    const T* input1,
    const T* input2,
    T* output,
    size_t n,
    Func func,
    cudaStream_t stream = nullptr) {
    
    if (n == 0) return;
    
    constexpr int block_size = 256;
    constexpr int vec_size = optimal_vec_size<T>();
    
    int grid_size = static_cast<int>((n + block_size * vec_size - 1) / (block_size * vec_size));
    grid_size = std::min(grid_size, 65535);
    
    elementwise_binary_kernel<T, Func, vec_size><<<grid_size, block_size, 0, stream>>>(
        input1, input2, output, n, func);
    
    TC_CUDA_CHECK_LAST();
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// ReLU activation
template<typename T>
void relu(const T* input, T* output, size_t n, cudaStream_t stream = nullptr) {
    launch_elementwise(input, output, n, ReLU{}, stream);
}

/// SiLU activation
template<typename T>
void silu(const T* input, T* output, size_t n, cudaStream_t stream = nullptr) {
    launch_elementwise(input, output, n, SiLU{}, stream);
}

/// GeLU activation
template<typename T>
void gelu(const T* input, T* output, size_t n, cudaStream_t stream = nullptr) {
    launch_elementwise(input, output, n, GeLU{}, stream);
}

/// Sigmoid activation
template<typename T>
void sigmoid(const T* input, T* output, size_t n, cudaStream_t stream = nullptr) {
    launch_elementwise(input, output, n, Sigmoid{}, stream);
}

/// Tanh activation
template<typename T>
void tanh_activation(const T* input, T* output, size_t n, cudaStream_t stream = nullptr) {
    launch_elementwise(input, output, n, Tanh{}, stream);
}

/// Vector addition
template<typename T>
void vector_add(const T* a, const T* b, T* c, size_t n, cudaStream_t stream = nullptr) {
    launch_elementwise_binary(a, b, c, n, Add{}, stream);
}

/// Vector multiplication
template<typename T>
void vector_mul(const T* a, const T* b, T* c, size_t n, cudaStream_t stream = nullptr) {
    launch_elementwise_binary(a, b, c, n, Mul{}, stream);
}

} // namespace kernels
} // namespace tensorcraft
