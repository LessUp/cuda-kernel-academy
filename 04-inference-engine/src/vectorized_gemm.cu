#include "vectorized_gemm.cuh"

namespace mini_inference {

// Explicit instantiation
template __global__ void vectorized_gemm<128, 128, 8, 8, 8>(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K);

void launch_vectorized_gemm(const float* A, const float* B, float* C,
                            int M, int N, int K, cudaStream_t stream) {
    using Config = VectorizedGemmConfig;
    
    dim3 block(Config::THREADS_X, Config::THREADS_Y);
    dim3 grid((N + Config::BN - 1) / Config::BN, 
              (M + Config::BM - 1) / Config::BM);
    
    vectorized_gemm<Config::BM, Config::BN, Config::BK, Config::TM, Config::TN>
        <<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    
    CUDA_CHECK(cudaGetLastError());
}

} // namespace mini_inference
