#pragma once

#include "common.h"

namespace mini_inference {

// ============================================================================
// Vectorized GEMM with float4 loads
// Achieves higher memory throughput through 128-bit loads
// ============================================================================

// Configuration for vectorized GEMM
struct VectorizedGemmConfig {
    static constexpr int BM = 128;      // Block tile M
    static constexpr int BN = 128;      // Block tile N
    static constexpr int BK = 8;        // Block tile K
    static constexpr int TM = 8;        // Thread tile M
    static constexpr int TN = 8;        // Thread tile N
    static constexpr int THREADS_X = BN / TN;  // 16
    static constexpr int THREADS_Y = BM / TM;  // 16
    static constexpr int NUM_THREADS = THREADS_X * THREADS_Y;  // 256
};

// Vectorized load helper
__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

// Vectorized store helper
__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

// Warp-level matrix multiply accumulate (WMMA-like pattern)
template<int TM, int TN>
__device__ __forceinline__ void warp_mma(
    float regA[TM], float regB[TN], float regC[TM][TN]) {
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            regC[m][n] = fmaf(regA[m], regB[n], regC[m][n]);
        }
    }
}

// Vectorized GEMM kernel with float4 loads
template<int BM, int BN, int BK, int TM, int TN>
__global__ void vectorized_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

    // Shared memory with padding to avoid bank conflicts
    __shared__ float As[BK][BM + 4];
    __shared__ float Bs[BK][BN + 4];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int THREADS_X = BN / TN;
    const int THREADS_Y = BM / TM;
    const int NUM_THREADS = THREADS_X * THREADS_Y;
    const int tid = ty * THREADS_X + tx;

    // Starting position for this thread's output tile
    const int row_start = by * BM + ty * TM;
    const int col_start = bx * BN + tx * TN;

    // Register storage
    float regC[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    const int num_tiles = (K + BK - 1) / BK;

    for (int tile = 0; tile < num_tiles; tile++) {
        // Cooperative loading of A tile (transposed storage)
        // Each thread loads multiple elements
        #pragma unroll
        for (int i = 0; i < (BM * BK) / NUM_THREADS; i++) {
            int idx = tid + i * NUM_THREADS;
            int a_k = idx % BK;
            int a_m = idx / BK;
            int global_row = by * BM + a_m;
            int global_col = tile * BK + a_k;

            if (global_row < M && global_col < K) {
                As[a_k][a_m] = A[global_row * K + global_col];
            } else {
                As[a_k][a_m] = 0.0f;
            }
        }

        // Cooperative loading of B tile
        #pragma unroll
        for (int i = 0; i < (BK * BN) / NUM_THREADS; i++) {
            int idx = tid + i * NUM_THREADS;
            int b_k = idx / BN;
            int b_n = idx % BN;
            int global_row = tile * BK + b_k;
            int global_col = bx * BN + b_n;

            if (global_row < K && global_col < N) {
                Bs[b_k][b_n] = B[global_row * N + global_col];
            } else {
                Bs[b_k][b_n] = 0.0f;
            }
        }

        __syncthreads();

        // Compute on this tile
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // Load A fragment
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                regA[m] = As[k][ty * TM + m];
            }

            // Load B fragment
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                regB[n] = Bs[k][tx * TN + n];
            }

            // Outer product using FMA
            warp_mma<TM, TN>(regA, regB, regC);
        }

        __syncthreads();
    }

    // Write results with vectorized stores where possible.
    // A float4 store requires the address to be 16B-aligned, i.e. the element
    // index (out_row*N + out_col) must be a multiple of 4. out_col is always a
    // multiple of 4, but out_row*N is only when N is a multiple of 4; for other
    // N a misaligned vector store triggers "misaligned address" on the device.
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int out_row = row_start + m;
        if (out_row < M) {
            bool row_aligned = (static_cast<size_t>(out_row) * N) % 4 == 0;
            #pragma unroll
            for (int n = 0; n < TN; n += 4) {
                int out_col = col_start + n;
                if (out_col + 3 < N && row_aligned) {
                    float4 result = make_float4(
                        regC[m][n], regC[m][n+1], regC[m][n+2], regC[m][n+3]);
                    store_float4(&C[out_row * N + out_col], result);
                } else {
                    // Handle boundary or non-16B-aligned rows with scalar stores
                    for (int nn = 0; nn < 4 && out_col + nn < N; nn++) {
                        C[out_row * N + out_col + nn] = regC[m][n + nn];
                    }
                }
            }
        }
    }
}

// Launch wrapper
void launch_vectorized_gemm(const float* A, const float* B, float* C,
                            int M, int N, int K, cudaStream_t stream = 0);

} // namespace mini_inference
