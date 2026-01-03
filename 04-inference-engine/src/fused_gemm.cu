#include "kernels.cuh"

namespace mini_inference {

// ============================================================================
// Level 6: Fused GEMM + Bias + ReLU
// Combines matrix multiplication with bias addition and ReLU activation
// in a single kernel to reduce memory traffic
// ============================================================================

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

template<int _BM, int _BN, int _BK, int _TM, int _TN, bool ADD_BIAS, bool APPLY_RELU>
__global__ void fused_gemm_bias_relu(const float* A, const float* B, float* C,
                                      const float* bias, int M, int N, int K) {
    __shared__ float As[_BK][_BM + 1];
    __shared__ float Bs[_BK][_BN + 1];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int THREADS_X = _BN / _TN;
    const int THREADS_Y = _BM / _TM;
    const int NUM_THREADS = THREADS_X * THREADS_Y;
    
    const int tid = ty * THREADS_X + tx;
    
    const int row_start = by * _BM + ty * _TM;
    const int col_start = bx * _BN + tx * _TN;
    
    float regC[_TM][_TN] = {0.0f};
    float regA[_TM];
    float regB[_TN];
    
    const int num_tiles = (K + _BK - 1) / _BK;
    const int A_ELEMENTS = (_BM * _BK) / NUM_THREADS;
    const int B_ELEMENTS = (_BK * _BN) / NUM_THREADS;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        #pragma unroll
        for (int i = 0; i < A_ELEMENTS; i++) {
            int idx = tid * A_ELEMENTS + i;
            int a_row = by * _BM + idx / _BK;
            int a_col = tile * _BK + idx % _BK;
            
            if (a_row < M && a_col < K) {
                As[idx % _BK][idx / _BK] = A[a_row * K + a_col];
            } else {
                As[idx % _BK][idx / _BK] = 0.0f;
            }
        }
        
        #pragma unroll
        for (int i = 0; i < B_ELEMENTS; i++) {
            int idx = tid * B_ELEMENTS + i;
            int b_row = tile * _BK + idx / _BN;
            int b_col = bx * _BN + idx % _BN;
            
            if (b_row < K && b_col < N) {
                Bs[idx / _BN][idx % _BN] = B[b_row * N + b_col];
            } else {
                Bs[idx / _BN][idx % _BN] = 0.0f;
            }
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < _BK; k++) {
            #pragma unroll
            for (int m = 0; m < _TM; m++) {
                regA[m] = As[k][ty * _TM + m];
            }
            
            #pragma unroll
            for (int n = 0; n < _TN; n++) {
                regB[n] = Bs[k][tx * _TN + n];
            }
            
            #pragma unroll
            for (int m = 0; m < _TM; m++) {
                #pragma unroll
                for (int n = 0; n < _TN; n++) {
                    regC[m][n] += regA[m] * regB[n];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Apply fusion operations and write results
    #pragma unroll
    for (int m = 0; m < _TM; m++) {
        int out_row = row_start + m;
        if (out_row < M) {
            #pragma unroll
            for (int n = 0; n < _TN; n++) {
                int out_col = col_start + n;
                if (out_col < N) {
                    float val = regC[m][n];
                    
                    // Add bias (compile-time branch)
                    if constexpr (ADD_BIAS) {
                        val += bias[out_col];
                    }
                    
                    // Apply ReLU (compile-time branch)
                    if constexpr (APPLY_RELU) {
                        val = fmaxf(val, 0.0f);
                    }
                    
                    C[out_row * N + out_col] = val;
                }
            }
        }
    }
}

// Explicit instantiations for all combinations
template __global__ void fused_gemm_bias_relu<BM, BN, BK, TM, TN, false, false>(
    const float* A, const float* B, float* C, const float* bias, int M, int N, int K);
template __global__ void fused_gemm_bias_relu<BM, BN, BK, TM, TN, true, false>(
    const float* A, const float* B, float* C, const float* bias, int M, int N, int K);
template __global__ void fused_gemm_bias_relu<BM, BN, BK, TM, TN, false, true>(
    const float* A, const float* B, float* C, const float* bias, int M, int N, int K);
template __global__ void fused_gemm_bias_relu<BM, BN, BK, TM, TN, true, true>(
    const float* A, const float* B, float* C, const float* bias, int M, int N, int K);

void launch_fused_gemm(const float* A, const float* B, float* C,
                       const float* bias, int M, int N, int K,
                       bool add_bias, bool apply_relu, cudaStream_t stream) {
    const int THREADS_X = BN / TN;
    const int THREADS_Y = BM / TM;
    
    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    // Select kernel based on fusion options
    if (add_bias && apply_relu) {
        fused_gemm_bias_relu<BM, BN, BK, TM, TN, true, true>
            <<<grid, block, 0, stream>>>(A, B, C, bias, M, N, K);
    } else if (add_bias) {
        fused_gemm_bias_relu<BM, BN, BK, TM, TN, true, false>
            <<<grid, block, 0, stream>>>(A, B, C, bias, M, N, K);
    } else if (apply_relu) {
        fused_gemm_bias_relu<BM, BN, BK, TM, TN, false, true>
            <<<grid, block, 0, stream>>>(A, B, C, bias, M, N, K);
    } else {
        fused_gemm_bias_relu<BM, BN, BK, TM, TN, false, false>
            <<<grid, block, 0, stream>>>(A, B, C, bias, M, N, K);
    }
    
    CUDA_CHECK(cudaGetLastError());
}

#undef BM
#undef BN
#undef BK
#undef TM
#undef TN

} // namespace mini_inference
