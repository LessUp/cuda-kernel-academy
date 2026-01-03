# Kernel Optimization Guide

This guide explains the optimization techniques used in TensorCraft-HPC kernels.

## GEMM Optimization Journey

### Level 1: Naive Implementation

The simplest GEMM computes one output element per thread:

```cpp
template<typename T>
__global__ void gemm_naive(const T* A, const T* B, T* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

**Problems:**
- Each thread reads K elements from A and K elements from B
- Total global memory reads: M * N * 2K
- Very low arithmetic intensity

### Level 2: Shared Memory Tiling

Load tiles into shared memory to reduce global memory access:

```cpp
template<typename T, int TILE_SIZE = 32>
__global__ void gemm_tiled(const T* A, const T* B, T* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Cooperative loading into shared memory
        As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}
```

**Improvements:**
- Each tile is loaded once, used TILE_SIZE times
- Reduces global memory reads by factor of TILE_SIZE
- Coalesced memory access patterns

### Level 3: Double Buffering

Overlap memory loads with computation:

```cpp
template<typename T, int TILE_SIZE = 32>
__global__ void gemm_double_buffer(const T* A, const T* B, T* C, int M, int N, int K) {
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];
    
    // Prefetch first tile
    As[0][ty][tx] = A[...];
    Bs[0][ty][tx] = B[...];
    __syncthreads();
    
    for (int t = 0; t < num_tiles; ++t) {
        int curr = t % 2;
        int next = (t + 1) % 2;
        
        // Prefetch next tile while computing current
        if (t + 1 < num_tiles) {
            As[next][ty][tx] = A[...];
            Bs[next][ty][tx] = B[...];
        }
        
        // Compute with current tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[curr][ty][k] * Bs[curr][k][tx];
        }
        __syncthreads();
    }
}
```

**Improvements:**
- Hides memory latency with computation
- Better utilization of memory bandwidth

### Level 4: Tensor Cores (WMMA)

Use hardware matrix multiply units:

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void gemm_wmma(const half* A, const half* B, float* C, int M, int N, int K) {
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    
    fill_fragment(c_frag, 0.0f);
    
    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(a_frag, A + warp_row * K + k, K);
        load_matrix_sync(b_frag, B + k * N + warp_col, N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    store_matrix_sync(C + warp_row * N + warp_col, c_frag, N, mem_row_major);
}
```

**Improvements:**
- Hardware-accelerated 16x16x16 matrix multiply
- Much higher throughput than FMA instructions

## Softmax Optimization

### Online Algorithm

Compute softmax in a single pass using running max and sum:

```cpp
// Traditional: 3 passes (find max, compute exp sum, normalize)
// Online: 2 passes (combined max+sum, normalize)

float m_prev = -INFINITY;
float l_prev = 0.0f;

for (int i = 0; i < n; ++i) {
    float x = input[i];
    
    if (x > m_prev) {
        // Rescale previous sum
        l_prev *= expf(m_prev - x);
        m_prev = x;
    }
    l_prev += expf(x - m_prev);
}

// Normalize
for (int i = 0; i < n; ++i) {
    output[i] = expf(input[i] - m_prev) / l_prev;
}
```

### Warp Shuffle Reduction

Use warp shuffle for efficient parallel reduction:

```cpp
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}
```

## Memory Access Optimization

### Vectorized Loads

Load multiple elements per thread for better bandwidth:

```cpp
template<typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector {
    T val[N];
};

// Load 4 floats at once (128 bits)
using Vec4 = AlignedVector<float, 4>;

Vec4 data = *reinterpret_cast<const Vec4*>(&input[idx]);
```

### Coalesced Access

Ensure adjacent threads access adjacent memory:

```cpp
// Good: Coalesced
output[threadIdx.x] = input[threadIdx.x];

// Bad: Strided
output[threadIdx.x * stride] = input[threadIdx.x * stride];
```

### Bank Conflict Avoidance

Add padding to shared memory to avoid bank conflicts:

```cpp
// Without padding: bank conflicts on column access
__shared__ float tile[32][32];

// With padding: no bank conflicts
__shared__ float tile[32][32 + 1];
```

## Attention Optimization (FlashAttention)

### Key Ideas

1. **Tiling**: Process Q, K, V in blocks that fit in shared memory
2. **Online Softmax**: Compute softmax incrementally across K/V blocks
3. **Recomputation**: Trade compute for memory by not storing attention matrix

```cpp
// For each Q block:
for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
    // Load K, V tiles to shared memory
    // Compute QK^T for this block
    // Update running max and sum (online softmax)
    // Accumulate output with rescaling
}
```

### Memory Savings

- Standard attention: O(N²) memory for attention matrix
- FlashAttention: O(N) memory, only store output

## Profiling Tips

### Nsight Compute Metrics

Key metrics to monitor:

1. **Memory Throughput**: % of peak bandwidth achieved
2. **Compute Throughput**: % of peak FLOPS achieved
3. **Occupancy**: Active warps / maximum warps
4. **Warp Stall Reasons**: Why warps are waiting

### Roofline Analysis

Plot your kernel on the roofline model:

- **Memory-bound**: Below the sloped line (increase arithmetic intensity)
- **Compute-bound**: Below the flat line (optimize compute)

### Common Bottlenecks

1. **Low occupancy**: Reduce register/shared memory usage
2. **Memory bandwidth**: Use vectorized loads, improve coalescing
3. **Bank conflicts**: Add padding to shared memory
4. **Warp divergence**: Minimize conditional branches
