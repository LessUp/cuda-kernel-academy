# GEMM 优化详解

本文档详细解释 Mini-Inference Engine 中实现的各种 GEMM 优化技术。

## 目录

1. [GEMM 基础](#gemm-基础)
2. [Level 1: Naive 实现](#level-1-naive-实现)
3. [Level 2: Tiled GEMM](#level-2-tiled-gemm)
4. [Level 3: Memory Coalescing](#level-3-memory-coalescing)
5. [Level 4: Double Buffering](#level-4-double-buffering)
6. [Level 5: Register Blocking](#level-5-register-blocking)
7. [Level 6: Kernel Fusion](#level-6-kernel-fusion)
8. [Level 7: Vectorized Loads](#level-7-vectorized-loads)
9. [性能对比](#性能对比)

## GEMM 基础

### 定义

GEMM (General Matrix Multiply) 计算:
```
C = α × A × B + β × C
```

简化版本 (α=1, β=0):
```
C = A × B
```

其中:
- A: M × K 矩阵
- B: K × N 矩阵
- C: M × N 矩阵

### 计算复杂度

- 浮点运算: `2 × M × N × K` FLOPs
- 内存访问: `(M×K + K×N + M×N) × sizeof(float)` bytes
- 计算强度: `2×M×N×K / (M×K + K×N + M×N) / 4` FLOP/byte

### 为什么 GEMM 重要?

神经网络中的主要计算:
- 全连接层: `Y = X × W + b`
- 卷积层 (im2col): 转换为 GEMM
- 注意力机制: `Attention = softmax(Q × K^T / √d) × V`

## Level 1: Naive 实现

### 算法

```cpp
__global__ void naive_matmul(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### 分析

**优点**:
- 实现简单，易于理解
- 正确性容易验证

**缺点**:
- 每个输出元素需要 2K 次全局内存访问
- 总内存访问: `2 × M × N × K` 次
- 内存带宽成为瓶颈

**性能**: ~5-10% of cuBLAS

### 内存访问模式

```
Thread (row, col) 访问:
A[row, 0], A[row, 1], ..., A[row, K-1]  (K 次)
B[0, col], B[1, col], ..., B[K-1, col]  (K 次)
C[row, col]                             (1 次)
```

## Level 2: Tiled GEMM

### 核心思想

将矩阵分成小块 (tiles)，加载到共享内存，减少全局内存访问。

### 算法

```cpp
#define TILE_SIZE 32

__global__ void tiled_gemm(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载 A tile 到共享内存
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        
        // 加载 B tile 到共享内存
        if (t * TILE_SIZE + threadIdx.y < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // 计算部分和
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 分析

**内存访问优化**:
```
原始: 2 × M × N × K 次全局内存访问
优化后: 2 × M × N × K / TILE_SIZE 次全局内存访问
```

**共享内存使用**:
```
2 × TILE_SIZE × TILE_SIZE × sizeof(float) = 2 × 32 × 32 × 4 = 8 KB
```

**性能**: ~20-30% of cuBLAS

### 图示

```
┌─────────────────────────────────────────────────────────────┐
│                        Matrix A (M × K)                      │
│  ┌─────────┐                                                │
│  │ A_tile  │ ──────────────────────────────────────────────▶│
│  │ 32 × 32 │                                                │
│  └─────────┘                                                │
└─────────────────────────────────────────────────────────────┘
                              ×
┌─────────────────────────────────────────────────────────────┐
│                        Matrix B (K × N)                      │
│  ┌─────────┐                                                │
│  │ B_tile  │                                                │
│  │ 32 × 32 │                                                │
│  └─────────┘                                                │
│      │                                                      │
│      ▼                                                      │
└─────────────────────────────────────────────────────────────┘
                              =
┌─────────────────────────────────────────────────────────────┐
│                        Matrix C (M × N)                      │
│  ┌─────────┐                                                │
│  │ C_tile  │ ◀── 累加多个 A_tile × B_tile                   │
│  │ 32 × 32 │                                                │
│  └─────────┘                                                │
└─────────────────────────────────────────────────────────────┘
```

## Level 3: Memory Coalescing

### 核心思想

确保同一 warp 中的线程访问连续的内存地址，最大化内存带宽利用率。

### 合并访问 vs 非合并访问

```
合并访问 (Coalesced):
Thread 0 → Address 0x1000
Thread 1 → Address 0x1004
Thread 2 → Address 0x1008
...
Thread 31 → Address 0x107C
→ 1 次 128 字节内存事务

非合并访问 (Strided):
Thread 0 → Address 0x1000
Thread 1 → Address 0x2000
Thread 2 → Address 0x3000
...
→ 32 次独立内存事务
```

### 优化技巧

```cpp
// 优化前: 列优先访问 B (非合并)
Bs[threadIdx.y][threadIdx.x] = B[k * N + col];

// 优化后: 行优先访问 B (合并)
// 通过转置 B 或调整访问模式
```

### 性能: ~30-40% of cuBLAS

## Level 4: Double Buffering

### 核心思想

使用两组共享内存缓冲区，在计算当前 tile 的同时预取下一个 tile。

### 算法

```cpp
__global__ void double_buffer_gemm(...) {
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];
    
    int buffer = 0;
    
    // 预取第一个 tile
    load_tile(As[0], Bs[0], 0);
    __syncthreads();
    
    for (int t = 0; t < num_tiles; t++) {
        // 异步加载下一个 tile
        if (t + 1 < num_tiles) {
            load_tile_async(As[1-buffer], Bs[1-buffer], t+1);
        }
        
        // 计算当前 tile
        compute_tile(As[buffer], Bs[buffer]);
        
        buffer = 1 - buffer;
        __syncthreads();
    }
}
```

### 时间线

```
Without Double Buffering:
|--Load--|--Compute--|--Load--|--Compute--|--Load--|--Compute--|

With Double Buffering:
|--Load 0--|
           |--Compute 0--|--Compute 1--|--Compute 2--|
              |--Load 1--|--Load 2--|
```

### 性能: ~40-50% of cuBLAS

## Level 5: Register Blocking

### 核心思想

每个线程计算多个输出元素，增加计算密度，减少共享内存访问。

### 算法

```cpp
template<int BM, int BN, int BK, int TM, int TN>
__global__ void optimized_gemm(...) {
    // 每个线程的寄存器
    float regA[TM];
    float regB[TN];
    float regC[TM][TN] = {0};
    
    for (int k = 0; k < K; k += BK) {
        // 加载到共享内存
        load_to_shared(As, Bs, k);
        __syncthreads();
        
        // 从共享内存加载到寄存器并计算
        for (int kk = 0; kk < BK; kk++) {
            // 加载 A 的 TM 个元素到寄存器
            for (int m = 0; m < TM; m++) {
                regA[m] = As[threadRow * TM + m][kk];
            }
            
            // 加载 B 的 TN 个元素到寄存器
            for (int n = 0; n < TN; n++) {
                regB[n] = Bs[kk][threadCol * TN + n];
            }
            
            // 计算 TM × TN 外积
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    regC[m][n] += regA[m] * regB[n];
                }
            }
        }
        __syncthreads();
    }
    
    // 写回结果
    store_result(regC);
}
```

### 参数选择

| 参数 | 典型值 | 说明 |
|------|--------|------|
| BM | 128 | Block 处理的 M 维度 |
| BN | 128 | Block 处理的 N 维度 |
| BK | 8 | 每次迭代的 K 维度 |
| TM | 8 | 每线程处理的 M 维度 |
| TN | 8 | 每线程处理的 N 维度 |

### 寄存器使用

```
每线程寄存器:
- regA[TM] = 8 个 float = 8 寄存器
- regB[TN] = 8 个 float = 8 寄存器
- regC[TM][TN] = 64 个 float = 64 寄存器
- 其他变量 ≈ 20 寄存器
总计: ~100 寄存器/线程
```

### 性能: ~70-80% of cuBLAS

## Level 6: Kernel Fusion

### 核心思想

将多个操作合并到一个 kernel 中，消除中间结果的内存读写。

### 融合操作

```
分离执行:
GEMM: C = A × B           → 写 C 到全局内存
Bias: C = C + bias        → 读 C, 写 C
ReLU: C = max(0, C)       → 读 C, 写 C
总计: 3 次读 + 3 次写

融合执行:
Fused: C = ReLU(A × B + bias)
总计: 0 次中间读写
```

### 实现

```cpp
template<bool ADD_BIAS, bool APPLY_RELU>
__global__ void fused_gemm_bias_relu(...) {
    // ... GEMM 计算 ...
    float result = gemm_result;
    
    if constexpr (ADD_BIAS) {
        result += bias[col];
    }
    
    if constexpr (APPLY_RELU) {
        result = fmaxf(0.0f, result);
    }
    
    C[row * N + col] = result;
}
```

### 性能提升

```
内存带宽节省: 2 × M × N × sizeof(float) bytes
对于 1024×1024 矩阵: 8 MB 内存带宽节省
```

## Level 7: Vectorized Loads

### 核心思想

使用 128 位向量加载 (float4) 减少内存事务数量。

### 实现

```cpp
// 标量加载: 4 次 32 位事务
float a = A[idx];
float b = A[idx+1];
float c = A[idx+2];
float d = A[idx+3];

// 向量加载: 1 次 128 位事务
float4 vec = *reinterpret_cast<float4*>(&A[idx]);
float a = vec.x;
float b = vec.y;
float c = vec.z;
float d = vec.w;
```

### 对齐要求

```cpp
// 地址必须 16 字节对齐
assert(reinterpret_cast<uintptr_t>(&A[idx]) % 16 == 0);
```

### 性能提升

- 减少 75% 的内存事务
- 更好的内存带宽利用率

## 性能对比

### 理论分析

| Level | 全局内存访问 | 共享内存访问 | 寄存器使用 |
|-------|-------------|-------------|-----------|
| Naive | 2MNK | 0 | 低 |
| Tiled | 2MNK/T | 2MNK | 低 |
| Register | 2MNK/T | 2MNK/(TM×TN) | 高 |

### 实测性能 (RTX 3060, 1024×1024×1024)

| Kernel | Time (ms) | GFLOPS | vs cuBLAS |
|--------|-----------|--------|-----------|
| cuBLAS | 0.34 | 6375 | 100% |
| Naive | 3.34 | 642 | 10% |
| Tiled | 2.41 | 891 | 14% |
| Coalesced | 3.25 | 661 | 10% |
| Double Buffer | 3.13 | 686 | 11% |
| Optimized | 0.82 | 2633 | 41% |
| Fused | 0.75 | 2867 | 45% |

### 优化建议

1. **小矩阵** (< 512): 使用较小的 block size
2. **中等矩阵** (512-2048): 使用标准配置
3. **大矩阵** (> 2048): 使用大 block size + 向量化
4. **批量小矩阵**: 使用 batched GEMM

## 进一步优化方向

1. **Tensor Core**: 使用 WMMA API
2. **异步拷贝**: 使用 `cp.async` (Ampere+)
3. **Warp 级优化**: 使用 warp shuffle
4. **多 GPU**: 数据并行
