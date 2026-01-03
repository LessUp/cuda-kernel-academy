#pragma once

#include "common.h"
#include <cuda_fp16.h>

namespace mini_inference {

// ============================================================================
// Half Precision (FP16) GEMM
// Provides 2x memory bandwidth and potential for Tensor Core usage
// ============================================================================

// Convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half(f);
}

// Convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// Vectorized half2 operations
__device__ __forceinline__ half2 load_half2(const half* ptr) {
    return *reinterpret_cast<const half2*>(ptr);
}

__device__ __forceinline__ void store_half2(half* ptr, half2 val) {
    *reinterpret_cast<half2*>(ptr) = val;
}

// FP16 GEMM kernel with accumulation in FP32
template<int BM, int BN, int BK, int TM, int TN>
__global__ void half_gemm(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K) {
    
    __shared__ half As[BK][BM + 8];  // Extra padding for half
    __shared__ half Bs[BK][BN + 8];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int THREADS_X = BN / TN;
    const int THREADS_Y = BM / TM;
    const int NUM_THREADS = THREADS_X * THREADS_Y;
    const int tid = ty * THREADS_X + tx;
    
    const int row_start = by * BM + ty * TM;
    const int col_start = bx * BN + tx * TN;
    
    // Accumulate in FP32 for precision
    float regC[TM][TN] = {0.0f};
    half regA[TM];
    half regB[TN];
    
    const int num_tiles = (K + BK - 1) / BK;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        // Load A tile
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
                As[a_k][a_m] = __float2half(0.0f);
            }
        }
        
        // Load B tile
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
                Bs[b_k][b_n] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                regA[m] = As[k][ty * TM + m];
            }
            
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                regB[n] = Bs[k][tx * TN + n];
            }
            
            // FMA in FP32
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    regC[m][n] = fmaf(__half2float(regA[m]), 
                                      __half2float(regB[n]), 
                                      regC[m][n]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results (convert back to half)
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int out_row = row_start + m;
        if (out_row < M) {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                int out_col = col_start + n;
                if (out_col < N) {
                    C[out_row * N + out_col] = __float2half(regC[m][n]);
                }
            }
        }
    }
}

// Mixed precision: FP16 input, FP32 output
template<int BM, int BN, int BK, int TM, int TN>
__global__ void mixed_precision_gemm(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    
    __shared__ half As[BK][BM + 8];
    __shared__ half Bs[BK][BN + 8];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int THREADS_X = BN / TN;
    const int THREADS_Y = BM / TM;
    const int NUM_THREADS = THREADS_X * THREADS_Y;
    const int tid = ty * THREADS_X + tx;
    
    const int row_start = by * BM + ty * TM;
    const int col_start = bx * BN + tx * TN;
    
    float regC[TM][TN] = {0.0f};
    half regA[TM];
    half regB[TN];
    
    const int num_tiles = (K + BK - 1) / BK;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        #pragma unroll
        for (int i = 0; i < (BM * BK) / NUM_THREADS; i++) {
            int idx = tid + i * NUM_THREADS;
            int a_k = idx % BK;
            int a_m = idx / BK;
            int global_row = by * BM + a_m;
            int global_col = tile * BK + a_k;
            
            As[a_k][a_m] = (global_row < M && global_col < K) 
                ? A[global_row * K + global_col] 
                : __float2half(0.0f);
        }
        
        #pragma unroll
        for (int i = 0; i < (BK * BN) / NUM_THREADS; i++) {
            int idx = tid + i * NUM_THREADS;
            int b_k = idx / BN;
            int b_n = idx % BN;
            int global_row = tile * BK + b_k;
            int global_col = bx * BN + b_n;
            
            Bs[b_k][b_n] = (global_row < K && global_col < N)
                ? B[global_row * N + global_col]
                : __float2half(0.0f);
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                regA[m] = As[k][ty * TM + m];
            }
            
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                regB[n] = Bs[k][tx * TN + n];
            }
            
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    regC[m][n] = fmaf(__half2float(regA[m]), 
                                      __half2float(regB[n]), 
                                      regC[m][n]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write FP32 results
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int out_row = row_start + m;
        if (out_row < M) {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                int out_col = col_start + n;
                if (out_col < N) {
                    C[out_row * N + out_col] = regC[m][n];
                }
            }
        }
    }
}

// Launch wrappers
void launch_half_gemm(const half* A, const half* B, half* C,
                      int M, int N, int K, cudaStream_t stream = 0);

void launch_mixed_precision_gemm(const half* A, const half* B, float* C,
                                  int M, int N, int K, cudaStream_t stream = 0);

// Utility: convert float array to half
void convert_float_to_half(const float* src, half* dst, size_t n, cudaStream_t stream = 0);

// Utility: convert half array to float
void convert_half_to_float(const half* src, float* dst, size_t n, cudaStream_t stream = 0);

} // namespace mini_inference
