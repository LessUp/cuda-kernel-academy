#pragma once
/**
 * @file type_traits.hpp
 * @brief Type traits and numeric conversion helpers for TensorCraft
 */

#include <type_traits>

#include "features.hpp"

namespace tensorcraft {

template <typename T>
inline constexpr bool is_half_v = std::is_same_v<T, __half>;

template <typename T>
inline constexpr bool is_bfloat16_v = std::is_same_v<T, __nv_bfloat16>;

#ifdef TC_HAS_FP8
template <typename T>
inline constexpr bool is_fp8_v =
    std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>;
#else
template <typename T>
inline constexpr bool is_fp8_v = false;
#endif

template <typename T>
inline constexpr bool is_float_v = std::is_same_v<T, float>;

template <typename T>
inline constexpr bool is_numeric_v =
    std::is_arithmetic_v<T> || is_half_v<T> || is_bfloat16_v<T> || is_fp8_v<T>;

template <typename T>
TC_HOST_DEVICE_INLINE float to_float(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, double>) {
        return static_cast<float>(value);
    } else if constexpr (std::is_integral_v<T>) {
        return static_cast<float>(value);
    } else if constexpr (is_half_v<T>) {
        return __half2float(value);
    } else if constexpr (is_bfloat16_v<T>) {
        return __bfloat162float(value);
#ifdef TC_HAS_FP8
    } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>) {
        return static_cast<float>(value);
#endif
    } else {
        return static_cast<float>(value);
    }
}

template <typename T>
TC_HOST_DEVICE_INLINE T from_float(float value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, double>) {
        return static_cast<double>(value);
    } else if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(value);
    } else if constexpr (is_half_v<T>) {
        return __float2half(value);
    } else if constexpr (is_bfloat16_v<T>) {
        return __float2bfloat16(value);
#ifdef TC_HAS_FP8
    } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>) {
        return T(value);
#endif
    } else {
        return static_cast<T>(value);
    }
}

}  // namespace tensorcraft
