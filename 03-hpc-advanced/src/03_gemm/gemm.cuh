#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <concepts>

namespace hpc::gemm {

enum class GemmOpt {
    Naive,            // Step 1: Global memory only
    SharedMemTiling,  // Step 2: Shared memory tiling
    DoubleBuffer,     // Step 3: Double buffering
    RegisterTiling,   // Step 4: Register tiling
    TensorCoreWMMA,   // Step 5: WMMA API
    TensorCoreMMA,    // Step 6: MMA PTX
    SoftwarePipeline  // Step 7: Software pipelining
};

// C = alpha * A * B + beta * C
template <typename T, GemmOpt Opt = GemmOpt::SharedMemTiling>
    requires std::is_same_v<T, float> || std::is_same_v<T, __half> || std::is_same_v<T, int8_t>
void gemm(const T* A, const T* B, T* C,
          int M, int N, int K,
          float alpha = 1.0f, float beta = 0.0f,
          cudaStream_t stream = nullptr);

// CUTLASS comparison wrapper
template <typename T>
void gemm_cutlass(const T* A, const T* B, T* C,
                  int M, int N, int K,
                  float alpha = 1.0f, float beta = 0.0f,
                  cudaStream_t stream = nullptr);

} // namespace hpc::gemm
