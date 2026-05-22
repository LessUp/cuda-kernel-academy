#pragma once
/**
 * @file cuda_check.hpp
 * @brief CUDA error handling utilities with exception-based reporting
 *
 * Provides macros and functions for checking CUDA runtime API errors.
 * All errors are reported via exceptions (throw), allowing callers to catch
 * and handle them appropriately.
 *
 * @example
 * @code
 * #include <cuda_academy/core/cuda_check.hpp>
 *
 * // Basic usage
 * CA_CUDA_CHECK(cudaMalloc(&ptr, size));
 *
 * // Check last kernel launch error
 * my_kernel<<<grid, block>>>(args);
 * CA_CUDA_CHECK_LAST();
 *
 * // cuBLAS operations
 * CA_CUBLAS_CHECK(cublasSgemm(handle, ...));
 * @endcode
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <stdexcept>
#include <string>

namespace cuda_academy {
namespace core {

// ============================================================================
// Exception Types
// ============================================================================

/**
 * @brief Exception thrown when a CUDA runtime API call fails
 */
class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t error, const char* expr, const char* file, int line)
        : std::runtime_error(build_message(error, expr, file, line)), error_(error) {}

    /// Get the CUDA error code
    [[nodiscard]] cudaError_t code() const noexcept { return error_; }

private:
    static std::string build_message(cudaError_t error, const char* expr, const char* file,
                                     int line) {
        return std::string(file) + ":" + std::to_string(line) + ": CUDA error: " + expr + " -> " +
               cudaGetErrorString(error);
    }

    cudaError_t error_;
};

/**
 * @brief Exception thrown when a cuBLAS API call fails
 */
class CublasError : public std::runtime_error {
public:
    CublasError(cublasStatus_t status, const char* expr, const char* file, int line)
        : std::runtime_error(build_message(status, expr, file, line)), status_(status) {}

    /// Get the cuBLAS status code
    [[nodiscard]] cublasStatus_t status() const noexcept { return status_; }

private:
    static std::string build_message(cublasStatus_t status, const char* expr, const char* file,
                                     int line) {
        return std::string(file) + ":" + std::to_string(line) + ": cuBLAS error: " + expr + " -> " +
               cublas_status_to_string(status);
    }

    static const char* cublas_status_to_string(cublasStatus_t status) {
        switch (status) {
            case CUBLAS_STATUS_SUCCESS:
                return "SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED:
                return "NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED:
                return "ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE:
                return "INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH:
                return "ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR:
                return "MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED:
                return "EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR:
                return "INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED:
                return "NOT_SUPPORTED";
            case CUBLAS_STATUS_LICENSE_ERROR:
                return "LICENSE_ERROR";
            default:
                return "UNKNOWN";
        }
    }

    cublasStatus_t status_;
};

/**
 * @brief Exception thrown when a cuRAND API call fails
 */
class CurandError : public std::runtime_error {
public:
    CurandError(curandStatus_t status, const char* expr, const char* file, int line)
        : std::runtime_error(build_message(status, expr, file, line)), status_(status) {}

    /// Get the cuRAND status code
    [[nodiscard]] curandStatus_t status() const noexcept { return status_; }

private:
    static std::string build_message(curandStatus_t status, const char* expr, const char* file,
                                     int line) {
        return std::string(file) + ":" + std::to_string(line) + ": cuRAND error: " + expr + " -> " +
               curand_status_to_string(status);
    }

    static const char* curand_status_to_string(curandStatus_t status) {
        switch (status) {
            case CURAND_STATUS_SUCCESS:
                return "SUCCESS";
            case CURAND_STATUS_VERSION_MISMATCH:
                return "VERSION_MISMATCH";
            case CURAND_STATUS_NOT_INITIALIZED:
                return "NOT_INITIALIZED";
            case CURAND_STATUS_ALLOCATION_FAILED:
                return "ALLOCATION_FAILED";
            case CURAND_STATUS_TYPE_ERROR:
                return "TYPE_ERROR";
            case CURAND_STATUS_OUT_OF_RANGE:
                return "OUT_OF_RANGE";
            case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
                return "LENGTH_NOT_MULTIPLE";
            case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
                return "DOUBLE_PRECISION_REQUIRED";
            case CURAND_STATUS_LAUNCH_FAILURE:
                return "LAUNCH_FAILURE";
            case CURAND_STATUS_PREEXISTING_FAILURE:
                return "PREEXISTING_FAILURE";
            case CURAND_STATUS_INITIALIZATION_FAILED:
                return "INITIALIZATION_FAILED";
            case CURAND_STATUS_ARCH_MISMATCH:
                return "ARCH_MISMATCH";
            case CURAND_STATUS_INTERNAL_ERROR:
                return "INTERNAL_ERROR";
            default:
                return "UNKNOWN";
        }
    }

    curandStatus_t status_;
};

// ============================================================================
// Check Functions
// ============================================================================

/**
 * @brief Check a CUDA runtime API call and throw on error
 * @param error The cudaError_t returned by the CUDA call
 * @param expr String representation of the expression
 * @param file Source file name
 * @param line Source line number
 * @throws CudaError if error != cudaSuccess
 */
inline void cuda_check(cudaError_t error, const char* expr, const char* file, int line) {
    if (error != cudaSuccess) {
        throw CudaError(error, expr, file, line);
    }
}

/**
 * @brief Check the last CUDA kernel launch error
 * @param file Source file name
 * @param line Source line number
 * @throws CudaError if the last kernel launch failed
 */
inline void cuda_check_last(const char* file, int line) {
    cuda_check(cudaGetLastError(), "cudaGetLastError()", file, line);
}

/**
 * @brief Synchronize device and check for errors
 * @param file Source file name
 * @param line Source line number
 * @throws CudaError if synchronization fails
 */
inline void cuda_sync_check(const char* file, int line) {
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", file, line);
}

/**
 * @brief Check a cuBLAS API call and throw on error
 * @param status The cublasStatus_t returned by the cuBLAS call
 * @param expr String representation of the expression
 * @param file Source file name
 * @param line Source line number
 * @throws CublasError if status != CUBLAS_STATUS_SUCCESS
 */
inline void cublas_check(cublasStatus_t status, const char* expr, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw CublasError(status, expr, file, line);
    }
}

/**
 * @brief Check a cuRAND API call and throw on error
 * @param status The curandStatus_t returned by the cuRAND call
 * @param expr String representation of the expression
 * @param file Source file name
 * @param line Source line number
 * @throws CurandError if status != CURAND_STATUS_SUCCESS
 */
inline void curand_check(curandStatus_t status, const char* expr, const char* file, int line) {
    if (status != CURAND_STATUS_SUCCESS) {
        throw CurandError(status, expr, file, line);
    }
}

}  // namespace core
}  // namespace cuda_academy

// ============================================================================
// Convenience Macros
// ============================================================================

/// Check a CUDA runtime API call and throw on error
#define CA_CUDA_CHECK(expr) ::cuda_academy::core::cuda_check((expr), #expr, __FILE__, __LINE__)

/// Check the last kernel launch error (call after kernel launches)
#define CA_CUDA_CHECK_LAST() ::cuda_academy::core::cuda_check_last(__FILE__, __LINE__)

/// Synchronize device and check for errors
#define CA_CUDA_SYNC_CHECK() ::cuda_academy::core::cuda_sync_check(__FILE__, __LINE__)

/// Check a cuBLAS API call and throw on error
#define CA_CUBLAS_CHECK(expr) ::cuda_academy::core::cublas_check((expr), #expr, __FILE__, __LINE__)

/// Check a cuRAND API call and throw on error
#define CA_CURAND_CHECK(expr) ::cuda_academy::core::curand_check((expr), #expr, __FILE__, __LINE__)
