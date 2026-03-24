#pragma once
/**
 * @file cuda_check.hpp
 * @brief Common CUDA error handling helpers for TensorCraft
 */

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace tensorcraft {
namespace core {

class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t error, const char* expr, const char* file, int line)
        : std::runtime_error(build_message(error, expr, file, line)), error_(error) {}

    cudaError_t code() const noexcept { return error_; }

private:
    static std::string build_message(cudaError_t error, const char* expr, const char* file,
                                     int line) {
        return std::string(file) + ":" + std::to_string(line) + ": CUDA call failed: " + expr +
               " -> " + cudaGetErrorString(error);
    }

    cudaError_t error_;
};

inline void cuda_check(cudaError_t error, const char* expr, const char* file, int line) {
    if (error != cudaSuccess) {
        throw CudaError(error, expr, file, line);
    }
}

inline void cuda_check_last(const char* file, int line) {
    cuda_check(cudaGetLastError(), "cudaGetLastError()", file, line);
}

inline void cuda_sync_check(const char* file, int line) {
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", file, line);
}

}  // namespace core
}  // namespace tensorcraft

#define TC_CUDA_CHECK(expr) ::tensorcraft::core::cuda_check((expr), #expr, __FILE__, __LINE__)
#define TC_CUDA_CHECK_LAST() ::tensorcraft::core::cuda_check_last(__FILE__, __LINE__)
#define TC_CUDA_SYNC_CHECK() ::tensorcraft::core::cuda_sync_check(__FILE__, __LINE__)
