#pragma once

/**
 * @file cuda_check.cuh
 * @brief CUDA error handling for HPC Advanced module
 *
 * Note: This module uses exit() on error for simplicity in educational contexts.
 * For production code or library usage, prefer the exception-based error handling
 * from cuda_academy: <cuda_academy/core/cuda_check.hpp>
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace hpc {

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

} // namespace hpc
