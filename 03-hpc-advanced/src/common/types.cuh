#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>
#include <concepts>

namespace hpc {

// Type traits for CUDA types
template <typename T>
struct is_half : std::false_type {};

template <>
struct is_half<__half> : std::true_type {};

template <typename T>
inline constexpr bool is_half_v = is_half<T>::value;

template <typename T>
struct is_bfloat16 : std::false_type {};

template <>
struct is_bfloat16<__nv_bfloat16> : std::true_type {};

template <typename T>
inline constexpr bool is_bfloat16_v = is_bfloat16<T>::value;

// Concept for floating point types (including CUDA types)
template <typename T>
concept FloatingPoint = std::is_floating_point_v<T> ||
                        is_half_v<T> ||
                        is_bfloat16_v<T>;

// Type conversion utilities
template <typename To, typename From>
__host__ __device__ inline To type_cast(From val) {
    if constexpr (std::is_same_v<To, From>) {
        return val;
    } else if constexpr (std::is_same_v<To, __half>) {
        return __float2half(static_cast<float>(val));
    } else if constexpr (std::is_same_v<From, __half>) {
        return static_cast<To>(__half2float(val));
    } else if constexpr (std::is_same_v<To, __nv_bfloat16>) {
        return __float2bfloat16(static_cast<float>(val));
    } else if constexpr (std::is_same_v<From, __nv_bfloat16>) {
        return static_cast<To>(__bfloat162float(val));
    } else {
        return static_cast<To>(val);
    }
}

// Accumulator type selection (use higher precision for accumulation)
template <typename T>
struct AccumulatorType {
    using type = T;
};

template <>
struct AccumulatorType<__half> {
    using type = float;
};

template <>
struct AccumulatorType<__nv_bfloat16> {
    using type = float;
};

template <typename T>
using accumulator_t = typename AccumulatorType<T>::type;

// Data type enumeration
enum class DataType {
    Float32,
    Float16,
    BFloat16,
    Int8,
    Int32,
    FP8_E4M3,
    FP8_E5M2
};

// Get size of data type
inline size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return 4;
        case DataType::Float16: return 2;
        case DataType::BFloat16: return 2;
        case DataType::Int8: return 1;
        case DataType::Int32: return 4;
        case DataType::FP8_E4M3: return 1;
        case DataType::FP8_E5M2: return 1;
        default: return 0;
    }
}

} // namespace hpc
