#include "half_gemm.cuh"

namespace mini_inference {

// Configuration
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

// Explicit instantiations
template __global__ void half_gemm<BM, BN, BK, TM, TN>(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K);

template __global__ void mixed_precision_gemm<BM, BN, BK, TM, TN>(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K);

void launch_half_gemm(const half* A, const half* B, half* C,
                      int M, int N, int K, cudaStream_t stream) {
    const int THREADS_X = BN / TN;
    const int THREADS_Y = BM / TM;
    
    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    half_gemm<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void launch_mixed_precision_gemm(const half* A, const half* B, float* C,
                                  int M, int N, int K, cudaStream_t stream) {
    const int THREADS_X = BN / TN;
    const int THREADS_Y = BM / TM;
    
    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    mixed_precision_gemm<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

// Conversion kernels
__global__ void float_to_half_kernel(const float* src, half* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

__global__ void half_to_float_kernel(const half* src, float* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

void convert_float_to_half(const float* src, half* dst, size_t n, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    float_to_half_kernel<<<grid_size, block_size, 0, stream>>>(src, dst, n);
    CUDA_CHECK(cudaGetLastError());
}

void convert_half_to_float(const half* src, float* dst, size_t n, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    half_to_float_kernel<<<grid_size, block_size, 0, stream>>>(src, dst, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace mini_inference
