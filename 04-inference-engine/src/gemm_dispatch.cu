#include "kernels.cuh"

#include "tensorcraft/kernels/gemm.hpp"

namespace mini_inference {

namespace {

void launch_tensorcraft_gemm(const float* A, const float* B, float* C,
                             int M, int N, int K,
                             tensorcraft::kernels::GemmVersion version,
                             cudaStream_t stream) {
    tensorcraft::kernels::launch_gemm(
        A, B, C, M, N, K,
        1.0f, 0.0f,
        version,
        stream);
}

}  // namespace

void launch_naive_matmul(const float* A, const float* B, float* C,
                         int M, int N, int K, cudaStream_t stream) {
    launch_tensorcraft_gemm(A, B, C, M, N, K,
                            tensorcraft::kernels::GemmVersion::Naive, stream);
}

void launch_tiled_gemm(const float* A, const float* B, float* C,
                       int M, int N, int K, cudaStream_t stream) {
    launch_tensorcraft_gemm(A, B, C, M, N, K,
                            tensorcraft::kernels::GemmVersion::Tiled, stream);
}

void launch_coalesced_gemm(const float* A, const float* B, float* C,
                           int M, int N, int K, cudaStream_t stream) {
    launch_tensorcraft_gemm(A, B, C, M, N, K,
                            tensorcraft::kernels::GemmVersion::Tiled, stream);
}

void launch_double_buffer_gemm(const float* A, const float* B, float* C,
                               int M, int N, int K, cudaStream_t stream) {
    launch_tensorcraft_gemm(A, B, C, M, N, K,
                            tensorcraft::kernels::GemmVersion::DoubleBuffer, stream);
}

}  // namespace mini_inference
