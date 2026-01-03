#pragma once
/**
 * @file aligned_vector.hpp
 * @brief Aligned vector types for vectorized memory access
 * 
 * Provides aligned vector types that enable efficient vectorized loads
 * and stores on GPU, improving memory bandwidth utilization.
 */

#include "../core/features.hpp"
#include <cstddef>

namespace tensorcraft {

/**
 * @brief Aligned vector type for vectorized memory access
 * 
 * @tparam T Element type
 * @tparam N Number of elements (must be power of 2)
 * 
 * The alignment is set to sizeof(T) * N to enable efficient
 * vectorized loads (e.g., LDS.128 for 16-byte loads).
 */
template<typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector {
    static_assert(N > 0 && (N & (N - 1)) == 0, "N must be a power of 2");
    static_assert(sizeof(T) * N <= 16, "Vector size must not exceed 16 bytes");
    
    T val[N];
    
    /// Number of elements
    static constexpr int size = N;
    
    /// Size in bytes
    static constexpr size_t byte_size = sizeof(T) * N;
    
    /// Element access
    TC_HOST_DEVICE_INLINE T& operator[](int i) { return val[i]; }
    TC_HOST_DEVICE_INLINE const T& operator[](int i) const { return val[i]; }
    
    /// Pointer access
    TC_HOST_DEVICE_INLINE T* data() { return val; }
    TC_HOST_DEVICE_INLINE const T* data() const { return val; }
    
    /// Fill with value
    TC_HOST_DEVICE_INLINE void fill(T value) {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            val[i] = value;
        }
    }
    
    /// Zero initialization
    TC_HOST_DEVICE_INLINE void zero() {
        fill(T(0));
    }
};

// ============================================================================
// Common Type Aliases
// ============================================================================

/// 2-element vector
template<typename T>
using Vec2 = AlignedVector<T, 2>;

/// 4-element vector
template<typename T>
using Vec4 = AlignedVector<T, 4>;

/// 8-element vector (for 1-byte types like int8)
template<typename T>
using Vec8 = AlignedVector<T, 8>;

// ============================================================================
// Specific Type Aliases
// ============================================================================

// Float vectors
using float2_t = Vec2<float>;
using float4_t = Vec4<float>;

// Half vectors
using half2_t = Vec2<__half>;
using half4_t = Vec4<__half>;
using half8_t = Vec8<__half>;

// BFloat16 vectors
using bfloat2_t = Vec2<__nv_bfloat16>;
using bfloat4_t = Vec4<__nv_bfloat16>;
using bfloat8_t = Vec8<__nv_bfloat16>;

// Integer vectors
using int2_t = Vec2<int>;
using int4_t = Vec4<int>;
using int8_v = Vec8<int8_t>;  // 8 x int8

// ============================================================================
// Vectorized Load/Store Utilities
// ============================================================================

/**
 * @brief Load aligned vector from memory
 */
template<typename T, int N>
TC_DEVICE_INLINE AlignedVector<T, N> load_vector(const T* ptr) {
    return *reinterpret_cast<const AlignedVector<T, N>*>(ptr);
}

/**
 * @brief Store aligned vector to memory
 */
template<typename T, int N>
TC_DEVICE_INLINE void store_vector(T* ptr, const AlignedVector<T, N>& vec) {
    *reinterpret_cast<AlignedVector<T, N>*>(ptr) = vec;
}

/**
 * @brief Check if pointer is aligned for vector access
 */
template<typename T, int N>
TC_HOST_DEVICE_INLINE bool is_aligned(const T* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) % (sizeof(T) * N) == 0;
}

/**
 * @brief Get optimal vector size for type T
 * 
 * Returns the largest vector size that fits in 16 bytes (LDS.128)
 */
template<typename T>
constexpr int optimal_vec_size() {
    if constexpr (sizeof(T) == 1) return 8;      // 8 x 1 byte = 8 bytes
    else if constexpr (sizeof(T) == 2) return 8; // 8 x 2 bytes = 16 bytes
    else if constexpr (sizeof(T) == 4) return 4; // 4 x 4 bytes = 16 bytes
    else if constexpr (sizeof(T) == 8) return 2; // 2 x 8 bytes = 16 bytes
    else return 1;
}

// ============================================================================
// Vector Arithmetic Operations
// ============================================================================

/**
 * @brief Element-wise addition
 */
template<typename T, int N>
TC_DEVICE_INLINE AlignedVector<T, N> operator+(
    const AlignedVector<T, N>& a, 
    const AlignedVector<T, N>& b) {
    AlignedVector<T, N> result;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

/**
 * @brief Element-wise subtraction
 */
template<typename T, int N>
TC_DEVICE_INLINE AlignedVector<T, N> operator-(
    const AlignedVector<T, N>& a, 
    const AlignedVector<T, N>& b) {
    AlignedVector<T, N> result;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

/**
 * @brief Element-wise multiplication
 */
template<typename T, int N>
TC_DEVICE_INLINE AlignedVector<T, N> operator*(
    const AlignedVector<T, N>& a, 
    const AlignedVector<T, N>& b) {
    AlignedVector<T, N> result;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

/**
 * @brief Scalar multiplication
 */
template<typename T, int N>
TC_DEVICE_INLINE AlignedVector<T, N> operator*(
    const AlignedVector<T, N>& a, 
    T scalar) {
    AlignedVector<T, N> result;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        result[i] = a[i] * scalar;
    }
    return result;
}

/**
 * @brief Fused multiply-add: a * b + c
 */
template<typename T, int N>
TC_DEVICE_INLINE AlignedVector<T, N> fma(
    const AlignedVector<T, N>& a,
    const AlignedVector<T, N>& b,
    const AlignedVector<T, N>& c) {
    AlignedVector<T, N> result;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
    return result;
}

} // namespace tensorcraft
