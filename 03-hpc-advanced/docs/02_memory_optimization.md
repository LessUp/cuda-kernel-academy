# 访存优化详解

本文档详细介绍 CUDA 访存优化技术，包括合并访问、向量化加载和 Shared Memory 使用。

## 1. 合并访问 (Coalesced Access)

### 什么是合并访问？

当一个 Warp (32 个线程) 访问连续的内存地址时，GPU 可以将这些访问合并为一次或少数几次内存事务。

### 好的访问模式

```cpp
// ✓ 合并访问: 相邻线程访问相邻地址
__global__ void good_access(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;  // 线程 0 访问 data[0], 线程 1 访问 data[1], ...
    }
}
```

### 坏的访问模式

```cpp
// ✗ 非合并访问: 跨步访问
__global__ void bad_access(float* data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * stride < n) {
        data[idx * stride] = data[idx * stride] * 2.0f;  // 线程 0 访问 data[0], 线程 1 访问 data[stride], ...
    }
}
```

### 内存事务对比

```
合并访问 (stride=1):
Thread 0  → data[0]   ┐
Thread 1  → data[1]   │
Thread 2  → data[2]   ├─ 1 次 128B 事务
...                   │
Thread 31 → data[31]  ┘

非合并访问 (stride=32):
Thread 0  → data[0]    → 1 次 32B 事务
Thread 1  → data[32]   → 1 次 32B 事务
Thread 2  → data[64]   → 1 次 32B 事务
...
Thread 31 → data[992]  → 1 次 32B 事务
                       = 32 次事务！
```

---

## 2. 向量化加载 (Vectorized Load/Store)

### 为什么使用向量化？

- 减少指令数量
- 提高内存带宽利用率
- 更好的指令级并行

### float4 向量化示例

```cpp
// 标量版本
__global__ void relu_scalar(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// 向量化版本 (float4)
__global__ void relu_vectorized(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t vec_idx = idx * 4;
    
    if (vec_idx + 3 < n) {
        // 一次加载 4 个 float
        float4 in = reinterpret_cast<const float4*>(input)[idx];
        
        // 处理
        float4 out;
        out.x = fmaxf(0.0f, in.x);
        out.y = fmaxf(0.0f, in.y);
        out.z = fmaxf(0.0f, in.z);
        out.w = fmaxf(0.0f, in.w);
        
        // 一次存储 4 个 float
        reinterpret_cast<float4*>(output)[idx] = out;
    }
}
```

### 性能对比

| 版本 | 指令数 | 带宽利用率 |
|------|--------|------------|
| 标量 | 4× | ~60% |
| float4 | 1× | ~90% |

### 对齐要求

```cpp
// float4 需要 16 字节对齐
float* data;
cudaMalloc(&data, n * sizeof(float));  // 默认 256 字节对齐，满足要求

// 检查对齐
assert(reinterpret_cast<uintptr_t>(data) % 16 == 0);
```

---

## 3. Grid Stride Loop

### 为什么使用 Grid Stride Loop？

- 处理任意大小的输入
- 更好的负载均衡
- 减少 Kernel 启动开销

### 实现

```cpp
__global__ void relu_grid_stride(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // 每个线程处理多个元素
    for (size_t i = idx; i < n; i += stride) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

// 启动配置
int block_size = 256;
int num_blocks = min((n + block_size - 1) / block_size, 1024);  // 限制 block 数量
relu_grid_stride<<<num_blocks, block_size>>>(input, output, n);
```

### 工作分配示意

```
n = 10000, block_size = 256, num_blocks = 4

Thread 0:    处理 0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216
Thread 1:    处理 1, 1025, 2049, 3073, 4097, 5121, 6145, 7169, 8193, 9217
...
Thread 1023: 处理 1023, 2047, 3071, 4095, 5119, 6143, 7167, 8191, 9215, ...
```

---

## 4. Shared Memory 优化

### Bank Conflict

Shared Memory 被分为 32 个 Bank，每个 Bank 宽度为 4 字节。

```
Bank 0:  addr 0,  32,  64,  96, ...
Bank 1:  addr 4,  36,  68, 100, ...
Bank 2:  addr 8,  40,  72, 104, ...
...
Bank 31: addr 124, 156, 188, 220, ...
```

### Bank Conflict 示例

```cpp
// ✗ Bank Conflict: 所有线程访问同一个 Bank
__shared__ float smem[32][32];
float val = smem[threadIdx.x][0];  // 所有线程访问 Bank 0

// ✓ 无 Bank Conflict: 每个线程访问不同 Bank
float val = smem[0][threadIdx.x];  // 线程 i 访问 Bank i
```

### Padding 消除 Bank Conflict

```cpp
// 矩阵转置中的 Bank Conflict
__shared__ float tile[32][32];  // 列访问时有 Bank Conflict

// 添加 Padding
__shared__ float tile[32][32 + 1];  // +1 消除 Bank Conflict

// 原理:
// 无 Padding: tile[0][0], tile[1][0], tile[2][0], ... 都在 Bank 0
// 有 Padding: tile[0][0] 在 Bank 0
//            tile[1][0] 在 Bank 1 (因为每行多了 1 个元素)
//            tile[2][0] 在 Bank 2
//            ...
```

---

## 5. 矩阵转置优化

### Naive 实现 (非合并写入)

```cpp
__global__ void transpose_naive(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        // 读: 合并访问 (行优先)
        // 写: 非合并访问 (列优先)
        output[col * rows + row] = input[row * cols + col];
    }
}
```

### Shared Memory 优化

```cpp
constexpr int TILE_DIM = 32;

__global__ void transpose_shared(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 避免 Bank Conflict
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // 合并读取到 Shared Memory
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    __syncthreads();
    
    // 计算转置后的坐标
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // 合并写入到 Global Memory
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### 访问模式对比

```
Naive:
读: input[row * cols + col]     → 合并 ✓
写: output[col * rows + row]    → 非合并 ✗

Shared Memory:
读: input[y * cols + x]         → 合并 ✓
tile[ty][tx] = ...              → 无 Bank Conflict (Padding)
写: output[y * rows + x]        → 合并 ✓
... = tile[tx][ty]              → 无 Bank Conflict (Padding)
```

---

## 6. 性能测量

### 带宽计算

```cpp
// 理论带宽 (RTX 4090: 1008 GB/s)
float theoretical_bandwidth = 1008.0f;  // GB/s

// 实际带宽
float data_size = n * sizeof(float) * 2;  // 读 + 写
float time_ms = timer.elapsed();
float actual_bandwidth = data_size / (time_ms * 1e6);  // GB/s

// 带宽利用率
float efficiency = actual_bandwidth / theoretical_bandwidth * 100.0f;
printf("Bandwidth: %.2f GB/s (%.1f%%)\n", actual_bandwidth, efficiency);
```

### Nsight Compute 分析

```bash
# 分析内存访问模式
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
              l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
    ./your_kernel

# 分析 Shared Memory Bank Conflict
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
              l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    ./your_kernel
```

---

## 7. 最佳实践总结

| 优化技术 | 适用场景 | 预期提升 |
|----------|----------|----------|
| 合并访问 | 所有 Kernel | 2-10× |
| 向量化 (float4) | Elementwise 操作 | 1.5-2× |
| Grid Stride Loop | 大数据量 | 1.2-1.5× |
| Shared Memory | 数据复用 | 2-5× |
| Padding | 消除 Bank Conflict | 1.2-1.5× |

## 参考资料

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Memory Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)
