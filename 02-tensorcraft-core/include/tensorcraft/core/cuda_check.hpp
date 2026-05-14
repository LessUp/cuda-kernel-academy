#pragma once
/**
 * @file cuda_check.hpp
 * @brief Common CUDA error handling helpers for TensorCraft
 *
 * This file provides backward-compatible aliases for the common
 * cuda_academy error handling utilities. New code should use
 * the CA_* macros from <cuda_academy/core/cuda_check.hpp>.
 */

#include <cuda_academy/core/cuda_check.hpp>

namespace tensorcraft {
namespace core {

// Backward-compatible type aliases
using CudaError = cuda_academy::core::CudaError;

// Backward-compatible inline functions (deprecated)
// These delegate to cuda_academy implementations
[[deprecated("Use CA_CUDA_CHECK from cuda_academy instead")]] inline void cuda_check(
    cudaError_t error, const char* expr, const char* file, int line) {
    cuda_academy::core::cuda_check(error, expr, file, line);
}

[[deprecated("Use CA_CUDA_CHECK_LAST from cuda_academy instead")]] inline void cuda_check_last(
    const char* file, int line) {
    cuda_academy::core::cuda_check_last(file, line);
}

[[deprecated("Use CA_CUDA_SYNC_CHECK from cuda_academy instead")]] inline void cuda_sync_check(
    const char* file, int line) {
    cuda_academy::core::cuda_sync_check(file, line);
}

}  // namespace core
}  // namespace tensorcraft

// Backward-compatible macros (deprecated)
#define TC_CUDA_CHECK(expr) CA_CUDA_CHECK(expr)
#define TC_CUDA_CHECK_LAST() CA_CUDA_CHECK_LAST()
#define TC_CUDA_SYNC_CHECK() CA_CUDA_SYNC_CHECK()
