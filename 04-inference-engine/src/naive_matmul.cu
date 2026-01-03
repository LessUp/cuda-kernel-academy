#include "kernels.cuh"

#include "tensorcraft/kernels/gemm.hpp"

namespace mini_inference {

// ============================================================================
// Level 1: Naive MatMul
// Each thread computes one element of the output matrix C
// ============================================================================

__global__ void naive_matmul(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launch_naive_matmul(const float* A, const float* B, float* C,
                         int M, int N, int K, cudaStream_t stream) {
    tensorcraft::kernels::launch_gemm(
        A, B, C, M, N, K,
        1.0f, 0.0f,
        tensorcraft::kernels::GemmVersion::Naive,
        stream);
}

} // namespace mini_inference
