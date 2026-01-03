# GEMM 优化详解

本文档详细介绍 GEMM (General Matrix Multiplication) 的 7 步优化路径。

## 概述

GEMM 计算: `C = α * A × B + β * C`

其中:
- A: M × K 矩阵
- B: K × N 矩阵
- C: M × N 矩阵
- α, β: 标量系数

## Step 1: Naive Global Memory

### 实现思路

每个线程计算输出矩阵 C 的一个元素。

```cpp
__global__ void gemm_naive_kernel(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
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

### 性能分析

- **问题**: 每个元素需要 2K 次全局内存访问
- **带宽利用率**: ~5-10%
- **TFLOPS**: ~0.5 (FP32, RTX 4090)

### 内存访问模式

```
Thread (0,0): A[0,0], A[0,1], ..., A[0,K-1]  ← 连续访问 ✓
              B[0,0], B[1,0], ..., B[K-1,0]  ← 跨步访问 ✗
```

---

## Step 2: Shared Memory Tiling

### 优化思路

将 A 和 B 的子块加载到 Shared Memory，减少全局内存访问。

```cpp
constexpr int TILE_SIZE = 32;

__global__ void gemm_shared_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 协作加载 Tile 到 Shared Memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // 计算部分点积
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 性能提升

- **全局内存访问减少**: K → K/TILE_SIZE
- **带宽利用率**: ~30-40%
- **TFLOPS**: ~2.0

### Tiling 示意图

```
        K                           K
    ┌───────┐                   ┌───────┐
    │       │                   │       │
M   │   A   │               K   │   B   │
    │       │                   │       │
    └───────┘                   └───────┘
        ↓                           ↓
    ┌───┬───┬───┐               ┌───┬───┬───┐
    │T1 │T2 │T3 │               │T1 │T2 │T3 │
    ├───┼───┼───┤               ├───┼───┼───┤
    │T4 │T5 │T6 │               │T4 │T5 │T6 │
    └───┴───┴───┘               └───┴───┴───┘
    
    每个 Block 处理一个 TILE_SIZE × TILE_SIZE 的输出块
```

---

## Step 3: Double Buffering

### 优化思路

使用双缓冲技术，在计算当前 Tile 的同时预取下一个 Tile。

```cpp
__global__ void gemm_double_buffer_kernel(const float* A, const float* B, float* C,
                                           int M, int N, int K) {
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];  // 双缓冲
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    int write_stage = 0;
    int read_stage = 0;

    // 预取第一个 Tile
    load_tile(As[write_stage], Bs[write_stage], A, B, 0, row, col, M, N, K);
    __syncthreads();

    for (int t = 0; t < num_tiles; ++t) {
        read_stage = write_stage;
        write_stage = 1 - write_stage;

        // 异步加载下一个 Tile
        if (t + 1 < num_tiles) {
            load_tile(As[write_stage], Bs[write_stage], A, B, t + 1, row, col, M, N, K);
        }

        // 计算当前 Tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[read_stage][threadIdx.y][k] * Bs[read_stage][k][threadIdx.x];
        }

        __syncthreads();
    }
    // ...
}
```

### 性能提升

- **隐藏内存延迟**: 计算与加载重叠
- **TFLOPS**: ~3.5

### 时间线对比

```
Without Double Buffering:
|--Load T1--|--Compute T1--|--Load T2--|--Compute T2--|

With Double Buffering:
|--Load T1--|--Compute T1--|--Compute T2--|--Compute T3--|
            |--Load T2----|--Load T3----|--Load T4----|
```

---

## Step 4: Register Tiling

### 优化思路

每个线程计算多个输出元素，减少 Shared Memory 访问。

```cpp
constexpr int REG_TILE_M = 8;  // 每个线程计算 8×8 个元素
constexpr int REG_TILE_N = 8;

__global__ void gemm_register_tiling_kernel(...) {
    // 寄存器累加器
    float reg_c[REG_TILE_M][REG_TILE_N] = {0.0f};

    for (int k_tile = 0; k_tile < K; k_tile += BLK_K) {
        // 加载到 Shared Memory
        // ...

        // 计算使用寄存器 Tiling
        for (int k = 0; k < BLK_K; ++k) {
            float reg_a[REG_TILE_M];
            float reg_b[REG_TILE_N];

            // 从 Shared Memory 加载到寄存器
            for (int m = 0; m < REG_TILE_M; ++m)
                reg_a[m] = As[k][thread_m * REG_TILE_M + m];
            for (int n = 0; n < REG_TILE_N; ++n)
                reg_b[n] = Bs[k][thread_n * REG_TILE_N + n];

            // 外积计算
            for (int m = 0; m < REG_TILE_M; ++m)
                for (int n = 0; n < REG_TILE_N; ++n)
                    reg_c[m][n] += reg_a[m] * reg_b[n];
        }
    }
    // ...
}
```

### 性能提升

- **Shared Memory 访问减少**: 8× (REG_TILE_M)
- **指令级并行**: 更多独立计算
- **TFLOPS**: ~6.0

---

## Step 5: Tensor Core (WMMA API)

### 优化思路

使用 NVIDIA Tensor Core 进行矩阵乘法，利用专用硬件加速。

```cpp
#include <mma.h>
using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void gemm_wmma_kernel(const __half* A, const __half* B, float* C,
                                  int M, int N, int K) {
    // 声明 Fragment
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        // 加载 Fragment
        wmma::load_matrix_sync(a_frag, A + row * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + col, N);

        // Tensor Core MMA
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 存储结果
    wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
}
```

### 性能提升

- **Tensor Core 吞吐**: 比 CUDA Core 高 8-16×
- **TFLOPS**: ~50+ (FP16)

### Tensor Core 架构

```
Tensor Core (每个 SM 有多个):
┌─────────────────────────────────────┐
│  16×16×16 Matrix Multiply-Accumulate │
│                                     │
│  A (16×16, FP16) × B (16×16, FP16)  │
│           ↓                         │
│     C (16×16, FP32)                 │
└─────────────────────────────────────┘
```

---

## Step 6: Tensor Core (MMA PTX)

### 优化思路

使用 PTX 指令直接控制 Tensor Core，获得更细粒度的控制。

```cpp
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
```

### 性能提升

- **更精细的寄存器控制**
- **TFLOPS**: ~60+

---

## Step 7: Software Pipelining

### 优化思路

使用多阶段流水线隐藏指令延迟。

```cpp
constexpr int PIPE_STAGES = 3;

__global__ void gemm_software_pipeline_kernel(...) {
    __shared__ float As[PIPE_STAGES][TILE_K][TILE_M + 1];
    __shared__ float Bs[PIPE_STAGES][TILE_K][TILE_N + 1];

    // Prologue: 填充流水线
    for (int stage = 0; stage < PIPE_STAGES - 1; ++stage) {
        load_tile(As[stage], Bs[stage], ...);
    }

    // Main loop: 流水线执行
    for (int k_tile = 0; k_tile < num_tiles; ++k_tile) {
        int compute_stage = k_tile % PIPE_STAGES;
        int load_stage = (k_tile + PIPE_STAGES - 1) % PIPE_STAGES;

        // 异步加载下一个 Tile
        if (k_tile + PIPE_STAGES - 1 < num_tiles) {
            load_tile(As[load_stage], Bs[load_stage], ...);
        }

        // 计算当前 Tile
        compute_tile(As[compute_stage], Bs[compute_stage], reg_c);
    }
}
```

### 性能提升

- **隐藏指令延迟**: 多阶段重叠
- **TFLOPS**: ~70+

### 流水线示意图

```
Stage 0: |--Load--|--Compute--|--Load--|--Compute--|
Stage 1:          |--Load--|--Compute--|--Load--|--Compute--|
Stage 2:                   |--Load--|--Compute--|--Load--|--Compute--|
```

---

## 性能对比总结

| Step | 优化技术 | TFLOPS (FP32) | 相对提升 |
|------|----------|---------------|----------|
| 1 | Naive | 0.5 | 1.0× |
| 2 | Shared Memory | 2.0 | 4.0× |
| 3 | Double Buffer | 3.5 | 7.0× |
| 4 | Register Tiling | 6.0 | 12.0× |
| 5 | WMMA | 50+ | 100× |
| 6 | MMA PTX | 60+ | 120× |
| 7 | Pipeline | 70+ | 140× |

## 参考资料

- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA Tensor Core Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
