#pragma once
/**
 * @file features.hpp
 * @brief Compiler, CUDA, and annotation helpers for TensorCraft
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef TC_HAS_FP8
    #include <cuda_fp8.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
    #define TC_DEVICE __device__
    #define TC_HOST __host__
    #define TC_HOST_DEVICE __host__ __device__
    #define TC_DEVICE_INLINE __device__ __forceinline__
    #define TC_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
    #define TC_DEVICE
    #define TC_HOST
    #define TC_HOST_DEVICE
    #define TC_DEVICE_INLINE inline
    #define TC_HOST_DEVICE_INLINE inline
#endif

#if defined(__CUDACC__) || defined(__GNUC__) || defined(__clang__)
    #define TC_RESTRICT __restrict__
#else
    #define TC_RESTRICT
#endif

namespace tensorcraft {
namespace core {

inline constexpr bool has_cuda_11 =
#ifdef TC_CUDA_11
    true;
#else
    false;
#endif

inline constexpr bool has_cuda_12 =
#ifdef TC_CUDA_12
    true;
#else
    false;
#endif

inline constexpr bool has_cuda_13 =
#ifdef TC_CUDA_13
    true;
#else
    false;
#endif

inline constexpr bool has_wmma =
#ifdef TC_HAS_WMMA
    true;
#else
    false;
#endif

inline constexpr bool has_tma =
#ifdef TC_HAS_TMA
    true;
#else
    false;
#endif

inline constexpr bool has_wgmma =
#ifdef TC_HAS_WGMMA
    true;
#else
    false;
#endif

inline constexpr bool has_fp8 =
#ifdef TC_HAS_FP8
    true;
#else
    false;
#endif

}  // namespace core
}  // namespace tensorcraft
