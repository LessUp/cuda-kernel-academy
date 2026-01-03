#include "kernels.cuh"

namespace mini_inference {

// ============================================================================
// Level 5: Register Blocked Optimized GEMM
// Each thread computes a TM x TN tile of output using register blocking
// Uses vectorized loads and avoids bank conflicts
// ============================================================================

// Default configuration optimized for modern GPUs
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

template<int _BM, int _BN, int _BK, int _TM, int _TN>
__global__ void optimized_gemm(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float As[_BK][_BM + 1];  // Transposed for coalesced access
    __shared__ float Bs[_BK][_BN + 1];
    
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Number of threads per block
    const int THREADS_X = _BN / _TN;  // 16
    const int THREADS_Y = _BM / _TM;  // 16
    const int NUM_THREADS = THREADS_X * THREADS_Y;  // 256
    
    // Linear thread ID
    const int tid = ty * THREADS_X + tx;
    
    // Starting position for this thread's output tile
    const int row_start = by * _BM + ty * _TM;
    const int col_start = bx * _BN + tx * _TN;
    
    // Register storage for output tile
    float regC[_TM][_TN] = {0.0f};
    
    // Register storage for A and B fragments
    float regA[_TM];
    float regB[_TN];
    
    // Number of tiles along K dimension
    const int num_tiles = (K + _BK - 1) / _BK;
    
    // Elements per thread for loading shared memory
    const int A_ELEMENTS = (_BM * _BK) / NUM_THREADS;
    const int B_ELEMENTS = (_BK * _BN) / NUM_THREADS;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        // Load A tile into shared memory (transposed)
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
        
        // Load B tile into shared memory
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
        
        // Compute on this tile
        #pragma unroll
        for (int k = 0; k < _BK; k++) {
            // Load A fragment from shared memory
            #pragma unroll
            for (int m = 0; m < _TM; m++) {
                regA[m] = As[k][ty * _TM + m];
            }
            
            // Load B fragment from shared memory
            #pragma unroll
            for (int n = 0; n < _TN; n++) {
                regB[n] = Bs[k][tx * _TN + n];
            }
            
            // Outer product
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
    
    // Write results to global memory
    #pragma unroll
    for (int m = 0; m < _TM; m++) {
        int out_row = row_start + m;
        if (out_row < M) {
            #pragma unroll
            for (int n = 0; n < _TN; n++) {
                int out_col = col_start + n;
                if (out_col < N) {
                    C[out_row * N + out_col] = regC[m][n];
                }
            }
        }
    }
}

// Explicit instantiation
template __global__ void optimized_gemm<BM, BN, BK, TM, TN>(
    const float* A, const float* B, float* C, int M, int N, int K);

void launch_optimized_gemm(const float* A, const float* B, float* C,
                           int M, int N, int K, cudaStream_t stream) {
    const int THREADS_X = BN / TN;  // 16
    const int THREADS_Y = BM / TM;  // 16
    
    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    optimized_gemm<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

#undef BM
#undef BN
#undef BK
#undef TM
#undef TN

} // namespace mini_inference
