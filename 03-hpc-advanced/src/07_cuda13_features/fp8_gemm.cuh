#pragma once

#include <cuda_runtime.h>

namespace hpc::cuda13 {

void fp8_gemm(const float* A, const float* B, float* C,
              int M, int N, int K,
              float scale_a = 1.0f, float scale_b = 1.0f,
              cudaStream_t stream = nullptr);

} // namespace hpc::cuda13
