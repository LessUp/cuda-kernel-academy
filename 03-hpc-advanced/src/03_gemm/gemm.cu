#include "gemm.cuh"
#include "../common/cuda_check.cuh"

namespace hpc::gemm {

constexpr int TILE_SIZE = 32;

// Naive GEMM: each thread computes one element
template <typename T>
__global__ void gemm_naive_kernel(const T* __restrict__ A,
                                   const T* __restrict__ B,
                                   T* __restrict__ C,
                                   int M, int N, int K,
                                   float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += static_cast<float>(A[row * K + k]) * static_cast<float>(B[k * N + col]);
        }
        C[row * N + col] = static_cast<T>(alpha * sum + beta * static_cast<float>(C[row * N + col]));
    }
}

// Shared memory tiling GEMM
template <typename T>
__global__ void gemm_shared_kernel(const T* __restrict__ A,
                                    const T* __restrict__ B,
                                    T* __restrict__ C,
                                    int M, int N, int K,
                                    float alpha, float beta) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = static_cast<float>(A[row * K + a_col]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = static_cast<float>(B[b_row * N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = static_cast<T>(alpha * sum + beta * static_cast<float>(C[row * N + col]));
    }
}

template <>
void gemm<float, GemmOpt::Naive>(const float* A, const float* B, float* C,
                                  int M, int N, int K,
                                  float alpha, float beta, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_naive_kernel<float><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}

template <>
void gemm<float, GemmOpt::SharedMemTiling>(const float* A, const float* B, float* C,
                                            int M, int N, int K,
                                            float alpha, float beta, cudaStream_t stream) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_shared_kernel<float><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}

template <>
void gemm<__half, GemmOpt::SharedMemTiling>(const __half* A, const __half* B, __half* C,
                                             int M, int N, int K,
                                             float alpha, float beta, cudaStream_t stream) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_shared_kernel<__half><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}

// Double buffering GEMM: overlap computation with memory loading
template <typename T>
__global__ void gemm_double_buffer_kernel(const T* __restrict__ A,
                                           const T* __restrict__ B,
                                           T* __restrict__ C,
                                           int M, int N, int K,
                                           float alpha, float beta) {
    // Double buffer: two sets of shared memory tiles
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    float sum = 0.0f;
    int write_stage = 0;
    int read_stage = 0;

    // Prefetch first tile
    {
        int a_col = threadIdx.x;
        int b_row = threadIdx.y;
        if (row < M && a_col < K) {
            As[write_stage][threadIdx.y][threadIdx.x] = static_cast<float>(A[row * K + a_col]);
        } else {
            As[write_stage][threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (b_row < K && col < N) {
            Bs[write_stage][threadIdx.y][threadIdx.x] = static_cast<float>(B[b_row * N + col]);
        } else {
            Bs[write_stage][threadIdx.y][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    for (int t = 0; t < num_tiles; ++t) {
        read_stage = write_stage;
        write_stage = 1 - write_stage;

        // Prefetch next tile while computing current
        if (t + 1 < num_tiles) {
            int next_t = t + 1;
            int a_col = next_t * TILE_SIZE + threadIdx.x;
            int b_row = next_t * TILE_SIZE + threadIdx.y;

            if (row < M && a_col < K) {
                As[write_stage][threadIdx.y][threadIdx.x] = static_cast<float>(A[row * K + a_col]);
            } else {
                As[write_stage][threadIdx.y][threadIdx.x] = 0.0f;
            }
            if (b_row < K && col < N) {
                Bs[write_stage][threadIdx.y][threadIdx.x] = static_cast<float>(B[b_row * N + col]);
            } else {
                Bs[write_stage][threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        // Compute using current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[read_stage][threadIdx.y][k] * Bs[read_stage][k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = static_cast<T>(alpha * sum + beta * static_cast<float>(C[row * N + col]));
    }
}

template <>
void gemm<float, GemmOpt::DoubleBuffer>(const float* A, const float* B, float* C,
                                         int M, int N, int K,
                                         float alpha, float beta, cudaStream_t stream) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_double_buffer_kernel<float><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}

template <>
void gemm<__half, GemmOpt::DoubleBuffer>(const __half* A, const __half* B, __half* C,
                                          int M, int N, int K,
                                          float alpha, float beta, cudaStream_t stream) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_double_buffer_kernel<__half><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}

} // namespace hpc::gemm


// Register tiling GEMM: each thread computes a small tile in registers
constexpr int REG_TILE_M = 8;  // Each thread computes 8x8 output elements
constexpr int REG_TILE_N = 8;
constexpr int BLK_M = 128;     // Block tile size
constexpr int BLK_N = 128;
constexpr int BLK_K = 8;

template <typename T>
__global__ void gemm_register_tiling_kernel(const T* __restrict__ A,
                                             const T* __restrict__ B,
                                             T* __restrict__ C,
                                             int M, int N, int K,
                                             float alpha, float beta) {
    // Thread block computes BLK_M x BLK_N output tile
    // Each thread computes REG_TILE_M x REG_TILE_N elements
    constexpr int THREADS_M = BLK_M / REG_TILE_M;  // 16 threads in M
    constexpr int THREADS_N = BLK_N / REG_TILE_N;  // 16 threads in N

    __shared__ float As[BLK_K][BLK_M];
    __shared__ float Bs[BLK_K][BLK_N];

    int thread_m = threadIdx.x / THREADS_N;
    int thread_n = threadIdx.x % THREADS_N;

    int block_row = blockIdx.y * BLK_M;
    int block_col = blockIdx.x * BLK_N;

    // Register tile for accumulation
    float reg_c[REG_TILE_M][REG_TILE_N] = {0.0f};

    for (int k_tile = 0; k_tile < K; k_tile += BLK_K) {
        // Cooperative loading of A and B tiles into shared memory
        // Each thread loads multiple elements
        for (int i = threadIdx.x; i < BLK_K * BLK_M; i += blockDim.x) {
            int k_idx = i / BLK_M;
            int m_idx = i % BLK_M;
            int global_m = block_row + m_idx;
            int global_k = k_tile + k_idx;
            if (global_m < M && global_k < K) {
                As[k_idx][m_idx] = static_cast<float>(A[global_m * K + global_k]);
            } else {
                As[k_idx][m_idx] = 0.0f;
            }
        }

        for (int i = threadIdx.x; i < BLK_K * BLK_N; i += blockDim.x) {
            int k_idx = i / BLK_N;
            int n_idx = i % BLK_N;
            int global_k = k_tile + k_idx;
            int global_n = block_col + n_idx;
            if (global_k < K && global_n < N) {
                Bs[k_idx][n_idx] = static_cast<float>(B[global_k * N + global_n]);
            } else {
                Bs[k_idx][n_idx] = 0.0f;
            }
        }

        __syncthreads();

        // Compute using register tiling
        #pragma unroll
        for (int k = 0; k < BLK_K; ++k) {
            // Load A fragment into registers
            float reg_a[REG_TILE_M];
            #pragma unroll
            for (int m = 0; m < REG_TILE_M; ++m) {
                reg_a[m] = As[k][thread_m * REG_TILE_M + m];
            }

            // Load B fragment into registers
            float reg_b[REG_TILE_N];
            #pragma unroll
            for (int n = 0; n < REG_TILE_N; ++n) {
                reg_b[n] = Bs[k][thread_n * REG_TILE_N + n];
            }

            // Outer product
            #pragma unroll
            for (int m = 0; m < REG_TILE_M; ++m) {
                #pragma unroll
                for (int n = 0; n < REG_TILE_N; ++n) {
                    reg_c[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory
    #pragma unroll
    for (int m = 0; m < REG_TILE_M; ++m) {
        #pragma unroll
        for (int n = 0; n < REG_TILE_N; ++n) {
            int global_m = block_row + thread_m * REG_TILE_M + m;
            int global_n = block_col + thread_n * REG_TILE_N + n;
            if (global_m < M && global_n < N) {
                float c_val = beta * static_cast<float>(C[global_m * N + global_n]);
                C[global_m * N + global_n] = static_cast<T>(alpha * reg_c[m][n] + c_val);
            }
        }
    }
}

template <>
void gemm<float, GemmOpt::RegisterTiling>(const float* A, const float* B, float* C,
                                           int M, int N, int K,
                                           float alpha, float beta, cudaStream_t stream) {
    constexpr int THREADS_PER_BLOCK = (BLK_M / REG_TILE_M) * (BLK_N / REG_TILE_N);
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + BLK_N - 1) / BLK_N, (M + BLK_M - 1) / BLK_M);
    gemm_register_tiling_kernel<float><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}

template <>
void gemm<__half, GemmOpt::RegisterTiling>(const __half* A, const __half* B, __half* C,
                                            int M, int N, int K,
                                            float alpha, float beta, cudaStream_t stream) {
    constexpr int THREADS_PER_BLOCK = (BLK_M / REG_TILE_M) * (BLK_N / REG_TILE_N);
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + BLK_N - 1) / BLK_N, (M + BLK_M - 1) / BLK_M);
    gemm_register_tiling_kernel<__half><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}


// Tensor Core GEMM using WMMA API
#include <mma.h>
using namespace nvcuda;

// WMMA tile dimensions (fixed by hardware)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Block tile dimensions for WMMA
constexpr int WMMA_BLK_M = 64;
constexpr int WMMA_BLK_N = 64;
constexpr int WMMA_BLK_K = 16;

__global__ void gemm_wmma_kernel(const __half* __restrict__ A,
                                  const __half* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K,
                                  float alpha, float beta) {
    // Warp-level matrix fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);

    // Calculate warp position
    int warp_id = threadIdx.x / 32;
    int warps_per_block_m = WMMA_BLK_M / WMMA_M;
    int warps_per_block_n = WMMA_BLK_N / WMMA_N;
    int warp_m = warp_id / warps_per_block_n;
    int warp_n = warp_id % warps_per_block_n;

    int block_row = blockIdx.y * WMMA_BLK_M;
    int block_col = blockIdx.x * WMMA_BLK_N;

    int row = block_row + warp_m * WMMA_M;
    int col = block_col + warp_n * WMMA_N;

    // Loop over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        if (row < M && k < K) {
            wmma::load_matrix_sync(a_frag, A + row * K + k, K);
        }
        if (k < K && col < N) {
            wmma::load_matrix_sync(b_frag, B + k * N + col, N);
        }

        // Perform matrix multiply-accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Scale and store result
    if (row < M && col < N) {
        // Apply alpha scaling
        for (int i = 0; i < c_frag.num_elements; ++i) {
            c_frag.x[i] = alpha * c_frag.x[i];
        }

        // Load existing C values and apply beta
        if (beta != 0.0f) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_old;
            wmma::load_matrix_sync(c_old, C + row * N + col, N, wmma::mem_row_major);
            for (int i = 0; i < c_frag.num_elements; ++i) {
                c_frag.x[i] += beta * c_old.x[i];
            }
        }

        wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
    }
}

// WMMA kernel for half precision output
__global__ void gemm_wmma_half_kernel(const __half* __restrict__ A,
                                       const __half* __restrict__ B,
                                       __half* __restrict__ C,
                                       int M, int N, int K,
                                       float alpha, float beta) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag;

    wmma::fill_fragment(c_frag, __float2half(0.0f));

    int warp_id = threadIdx.x / 32;
    int warps_per_block_n = WMMA_BLK_N / WMMA_N;
    int warp_m = warp_id / warps_per_block_n;
    int warp_n = warp_id % warps_per_block_n;

    int block_row = blockIdx.y * WMMA_BLK_M;
    int block_col = blockIdx.x * WMMA_BLK_N;

    int row = block_row + warp_m * WMMA_M;
    int col = block_col + warp_n * WMMA_N;

    for (int k = 0; k < K; k += WMMA_K) {
        if (row < M && k < K) {
            wmma::load_matrix_sync(a_frag, A + row * K + k, K);
        }
        if (k < K && col < N) {
            wmma::load_matrix_sync(b_frag, B + k * N + col, N);
        }
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    if (row < M && col < N) {
        wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
    }
}

template <>
void gemm<__half, GemmOpt::TensorCoreWMMA>(const __half* A, const __half* B, __half* C,
                                            int M, int N, int K,
                                            float alpha, float beta, cudaStream_t stream) {
    // Each block has multiple warps
    constexpr int WARPS_PER_BLOCK = (WMMA_BLK_M / WMMA_M) * (WMMA_BLK_N / WMMA_N);
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + WMMA_BLK_N - 1) / WMMA_BLK_N, (M + WMMA_BLK_M - 1) / WMMA_BLK_M);

    gemm_wmma_half_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}


// Tensor Core GEMM using MMA PTX instructions (more low-level control)
// This provides finer control over Tensor Core operations

// MMA PTX m16n8k16 for FP16
__device__ __forceinline__ void mma_m16n8k16_fp16(
    uint32_t* d, const uint32_t* a, const uint32_t* b, const uint32_t* c) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7}, "
        "{%8, %9};\n"
        : "=r"(d[0]), "=r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1])
    );
}

// Simplified MMA PTX kernel (demonstration)
__global__ void gemm_mma_ptx_kernel(const __half* __restrict__ A,
                                     const __half* __restrict__ B,
                                     __half* __restrict__ C,
                                     int M, int N, int K) {
    // This is a simplified demonstration of MMA PTX usage
    // Full implementation would require careful register management
    
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int block_row = blockIdx.y * 64;
    int block_col = blockIdx.x * 64;

    int warp_row = block_row + (warp_id / 2) * MMA_M;
    int warp_col = block_col + (warp_id % 2) * MMA_N * 4;

    // Accumulator registers
    uint32_t c_regs[2] = {0, 0};

    // Loop over K
    for (int k = 0; k < K; k += MMA_K) {
        // Load A fragment (simplified - actual implementation needs proper indexing)
        uint32_t a_regs[4];
        // Load B fragment
        uint32_t b_regs[2];

        // In a real implementation, we would load data here
        // For demonstration, we use WMMA as fallback
        
        // Perform MMA
        // mma_m16n8k16_fp16(c_regs, a_regs, b_regs, c_regs);
    }

    // Store results (simplified)
    // Actual implementation would write c_regs to global memory
}

template <>
void gemm<__half, GemmOpt::TensorCoreMMA>(const __half* A, const __half* B, __half* C,
                                           int M, int N, int K,
                                           float alpha, float beta, cudaStream_t stream) {
    // For now, fall back to WMMA implementation
    // Full MMA PTX implementation requires extensive register management
    gemm<__half, GemmOpt::TensorCoreWMMA>(A, B, C, M, N, K, alpha, beta, stream);
}


// Software Pipelining GEMM: hide memory latency with multi-stage pipeline
constexpr int PIPE_STAGES = 3;
constexpr int PIPE_TILE_M = 64;
constexpr int PIPE_TILE_N = 64;
constexpr int PIPE_TILE_K = 8;

template <typename T>
__global__ void gemm_software_pipeline_kernel(const T* __restrict__ A,
                                               const T* __restrict__ B,
                                               T* __restrict__ C,
                                               int M, int N, int K,
                                               float alpha, float beta) {
    // Multi-stage shared memory buffers
    __shared__ float As[PIPE_STAGES][PIPE_TILE_K][PIPE_TILE_M + 1];  // +1 for bank conflict avoidance
    __shared__ float Bs[PIPE_STAGES][PIPE_TILE_K][PIPE_TILE_N + 1];

    int block_row = blockIdx.y * PIPE_TILE_M;
    int block_col = blockIdx.x * PIPE_TILE_N;

    int thread_row = threadIdx.x / PIPE_TILE_N;
    int thread_col = threadIdx.x % PIPE_TILE_N;

    // Register accumulator
    float reg_c[4][4] = {0.0f};

    int num_k_tiles = (K + PIPE_TILE_K - 1) / PIPE_TILE_K;

    // Prologue: fill pipeline stages
    #pragma unroll
    for (int stage = 0; stage < PIPE_STAGES - 1 && stage < num_k_tiles; ++stage) {
        int k_offset = stage * PIPE_TILE_K;

        // Load A tile
        for (int i = threadIdx.x; i < PIPE_TILE_K * PIPE_TILE_M; i += blockDim.x) {
            int k_idx = i / PIPE_TILE_M;
            int m_idx = i % PIPE_TILE_M;
            int global_m = block_row + m_idx;
            int global_k = k_offset + k_idx;
            if (global_m < M && global_k < K) {
                As[stage][k_idx][m_idx] = static_cast<float>(A[global_m * K + global_k]);
            } else {
                As[stage][k_idx][m_idx] = 0.0f;
            }
        }

        // Load B tile
        for (int i = threadIdx.x; i < PIPE_TILE_K * PIPE_TILE_N; i += blockDim.x) {
            int k_idx = i / PIPE_TILE_N;
            int n_idx = i % PIPE_TILE_N;
            int global_k = k_offset + k_idx;
            int global_n = block_col + n_idx;
            if (global_k < K && global_n < N) {
                Bs[stage][k_idx][n_idx] = static_cast<float>(B[global_k * N + global_n]);
            } else {
                Bs[stage][k_idx][n_idx] = 0.0f;
            }
        }
    }

    __syncthreads();

    // Main loop with software pipelining
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        int compute_stage = k_tile % PIPE_STAGES;
        int load_stage = (k_tile + PIPE_STAGES - 1) % PIPE_STAGES;
        int next_k_tile = k_tile + PIPE_STAGES - 1;

        // Async load next tile (if available)
        if (next_k_tile < num_k_tiles) {
            int k_offset = next_k_tile * PIPE_TILE_K;

            for (int i = threadIdx.x; i < PIPE_TILE_K * PIPE_TILE_M; i += blockDim.x) {
                int k_idx = i / PIPE_TILE_M;
                int m_idx = i % PIPE_TILE_M;
                int global_m = block_row + m_idx;
                int global_k = k_offset + k_idx;
                if (global_m < M && global_k < K) {
                    As[load_stage][k_idx][m_idx] = static_cast<float>(A[global_m * K + global_k]);
                } else {
                    As[load_stage][k_idx][m_idx] = 0.0f;
                }
            }

            for (int i = threadIdx.x; i < PIPE_TILE_K * PIPE_TILE_N; i += blockDim.x) {
                int k_idx = i / PIPE_TILE_N;
                int n_idx = i % PIPE_TILE_N;
                int global_k = k_offset + k_idx;
                int global_n = block_col + n_idx;
                if (global_k < K && global_n < N) {
                    Bs[load_stage][k_idx][n_idx] = static_cast<float>(B[global_k * N + global_n]);
                } else {
                    Bs[load_stage][k_idx][n_idx] = 0.0f;
                }
            }
        }

        // Compute using current stage
        #pragma unroll
        for (int k = 0; k < PIPE_TILE_K; ++k) {
            float a_val[4], b_val[4];

            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                int m_idx = thread_row * 4 + m;
                a_val[m] = As[compute_stage][k][m_idx];
            }

            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                int n_idx = thread_col * 4 + n;
                b_val[n] = Bs[compute_stage][k][n_idx];
            }

            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    reg_c[m][n] += a_val[m] * b_val[n];
                }
            }
        }

        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            int global_m = block_row + thread_row * 4 + m;
            int global_n = block_col + thread_col * 4 + n;
            if (global_m < M && global_n < N) {
                float c_val = beta * static_cast<float>(C[global_m * N + global_n]);
                C[global_m * N + global_n] = static_cast<T>(alpha * reg_c[m][n] + c_val);
            }
        }
    }
}

template <>
void gemm<float, GemmOpt::SoftwarePipeline>(const float* A, const float* B, float* C,
                                             int M, int N, int K,
                                             float alpha, float beta, cudaStream_t stream) {
    constexpr int THREADS_PER_BLOCK = 256;
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + PIPE_TILE_N - 1) / PIPE_TILE_N, (M + PIPE_TILE_M - 1) / PIPE_TILE_M);
    gemm_software_pipeline_kernel<float><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}

template <>
void gemm<__half, GemmOpt::SoftwarePipeline>(const __half* A, const __half* B, __half* C,
                                              int M, int N, int K,
                                              float alpha, float beta, cudaStream_t stream) {
    constexpr int THREADS_PER_BLOCK = 256;
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + PIPE_TILE_N - 1) / PIPE_TILE_N, (M + PIPE_TILE_M - 1) / PIPE_TILE_M);
    gemm_software_pipeline_kernel<__half><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}


// Int8 GEMM implementation
template <>
__global__ void gemm_shared_kernel<int8_t>(const int8_t* __restrict__ A,
                                            const int8_t* __restrict__ B,
                                            int8_t* __restrict__ C,
                                            int M, int N, int K,
                                            float alpha, float beta) {
    __shared__ int As[TILE_SIZE][TILE_SIZE];
    __shared__ int Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int sum = 0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = static_cast<int>(A[row * K + a_col]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = static_cast<int>(B[b_row * N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float result = alpha * static_cast<float>(sum) + beta * static_cast<float>(C[row * N + col]);
        // Clamp to int8 range
        result = fmaxf(-128.0f, fminf(127.0f, result));
        C[row * N + col] = static_cast<int8_t>(result);
    }
}

template <>
void gemm<int8_t, GemmOpt::SharedMemTiling>(const int8_t* A, const int8_t* B, int8_t* C,
                                             int M, int N, int K,
                                             float alpha, float beta, cudaStream_t stream) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_shared_kernel<int8_t><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}

// Default implementations for other int8 optimization levels
template <>
void gemm<int8_t, GemmOpt::Naive>(const int8_t* A, const int8_t* B, int8_t* C,
                                   int M, int N, int K,
                                   float alpha, float beta, cudaStream_t stream) {
    gemm<int8_t, GemmOpt::SharedMemTiling>(A, B, C, M, N, K, alpha, beta, stream);
}

template <>
void gemm<int8_t, GemmOpt::DoubleBuffer>(const int8_t* A, const int8_t* B, int8_t* C,
                                          int M, int N, int K,
                                          float alpha, float beta, cudaStream_t stream) {
    gemm<int8_t, GemmOpt::SharedMemTiling>(A, B, C, M, N, K, alpha, beta, stream);
}
