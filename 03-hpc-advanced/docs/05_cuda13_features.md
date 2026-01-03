# CUDA 13 & Hopper 架构特性

本文档介绍 CUDA 13 和 Hopper/Blackwell 架构的新特性。

## 1. Hopper 架构概述

### 主要改进

| 特性 | Ampere (A100) | Hopper (H100) | 提升 |
|------|---------------|---------------|------|
| FP16 Tensor Core | 312 TFLOPS | 989 TFLOPS | 3.2× |
| FP8 Tensor Core | - | 1979 TFLOPS | 新增 |
| HBM 带宽 | 2 TB/s | 3.35 TB/s | 1.7× |
| L2 Cache | 40 MB | 50 MB | 1.25× |
| Shared Memory | 164 KB/SM | 228 KB/SM | 1.4× |

### 新特性列表

1. **TMA (Tensor Memory Accelerator)**: 异步数据搬运
2. **Thread Block Clusters**: Block 间协作
3. **Distributed Shared Memory**: 跨 Block 共享内存
4. **FP8 数据类型**: e4m3 和 e5m2
5. **Asynchronous Transaction Barrier**: 异步同步原语

---

## 2. TMA (Tensor Memory Accelerator)

### 什么是 TMA？

TMA 是 Hopper 架构引入的硬件单元，专门用于高效的多维数据搬运。

**优势**:
- 解放 SM 计算资源
- 自动处理边界条件
- 支持多维 Tensor 布局

### TMA 使用示例

```cpp
#include <cuda/barrier>
#include <cuda/pipeline>

// 创建 TMA 描述符
__host__ void create_tma_descriptor(
    CUtensorMap* desc,
    void* global_addr,
    uint64_t dims[5],
    uint64_t strides[5],
    uint32_t box_dims[5]) {
    
    cuTensorMapEncodeTiled(
        desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        4,  // 维度数
        global_addr,
        dims,
        strides,
        box_dims,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

// Kernel 中使用 TMA
__global__ void tma_copy_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    float* smem,
    int x, int y) {
    
    // 异步加载到 Shared Memory
    if (threadIdx.x == 0) {
        cuda::memcpy_async(
            smem,
            tensor_map,
            cuda::aligned_size_t<16>(sizeof(float) * TILE_SIZE * TILE_SIZE),
            cuda::pipeline<cuda::thread_scope_block>{}
        );
    }
    
    // 等待完成
    __syncthreads();
}
```

### TMA vs 传统加载

```
传统加载:
1. 每个线程计算地址
2. 每个线程发起加载请求
3. 合并访问优化
4. 线程等待数据

TMA 加载:
1. 一个线程发起 TMA 请求
2. TMA 硬件自动搬运
3. 其他线程可以继续计算
4. 通过 barrier 同步
```

---

## 3. Thread Block Clusters

### 什么是 Cluster？

Cluster 是多个 Thread Block 的组合，可以协作访问彼此的 Shared Memory。

```
传统模型:
Block 0 ←→ Shared Memory 0
Block 1 ←→ Shared Memory 1
Block 2 ←→ Shared Memory 2
(Block 间无法直接通信)

Cluster 模型:
┌─────────────────────────────────────┐
│           Cluster                   │
│  Block 0 ←→ Block 1 ←→ Block 2     │
│    ↕          ↕          ↕         │
│  SMEM 0 ←→ SMEM 1 ←→ SMEM 2       │
│     (Distributed Shared Memory)     │
└─────────────────────────────────────┘
```

### Cluster 配置

```cpp
// 编译时配置
__global__ __cluster_dims__(2, 2, 1)
void cluster_kernel(...) {
    // Cluster 大小: 2×2×1 = 4 个 Block
}

// 运行时配置
cudaLaunchConfig_t config;
config.gridDim = {num_blocks_x, num_blocks_y, 1};
config.blockDim = {256, 1, 1};

cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim = {2, 2, 1};

config.attrs = attrs;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, cluster_kernel, args...);
```

### Distributed Shared Memory

```cpp
__global__ __cluster_dims__(4, 1, 1)
void distributed_smem_kernel(float* output) {
    __shared__ float smem[256];
    
    // 获取 Cluster 信息
    cluster_group cluster = this_cluster();
    int cluster_rank = cluster.block_rank();
    int cluster_size = cluster.num_blocks();
    
    // 初始化本地 Shared Memory
    smem[threadIdx.x] = cluster_rank * 1000 + threadIdx.x;
    cluster.sync();
    
    // 访问其他 Block 的 Shared Memory
    int target_rank = (cluster_rank + 1) % cluster_size;
    float* remote_smem = cluster.map_shared_rank(smem, target_rank);
    
    float val = remote_smem[threadIdx.x];  // 读取远程 SMEM
    
    output[blockIdx.x * blockDim.x + threadIdx.x] = val;
}
```

### Cluster 归约

```cpp
__global__ __cluster_dims__(4, 1, 1)
void cluster_reduce_kernel(const float* input, float* output, int n) {
    __shared__ float smem[256];
    
    cluster_group cluster = this_cluster();
    
    // 每个 Block 计算局部和
    float local_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        local_sum += input[i];
    }
    
    // Block 内归约
    local_sum = block_reduce_sum(local_sum);
    
    if (threadIdx.x == 0) {
        smem[0] = local_sum;
    }
    cluster.sync();
    
    // Cluster 内归约 (使用 Distributed Shared Memory)
    if (threadIdx.x == 0 && cluster.block_rank() == 0) {
        float cluster_sum = 0.0f;
        for (int i = 0; i < cluster.num_blocks(); ++i) {
            float* remote_smem = cluster.map_shared_rank(smem, i);
            cluster_sum += remote_smem[0];
        }
        output[blockIdx.x / cluster.num_blocks()] = cluster_sum;
    }
}
```

---

## 4. FP8 数据类型

### FP8 格式

| 格式 | 指数位 | 尾数位 | 范围 | 精度 | 用途 |
|------|--------|--------|------|------|------|
| E4M3 | 4 | 3 | ±240 | 较高 | 权重、激活 |
| E5M2 | 5 | 2 | ±57344 | 较低 | 梯度 |

### FP8 GEMM

```cpp
#include <cuda_fp8.h>

__global__ void fp8_gemm_kernel(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    float* C,
    int M, int N, int K) {
    
    // 使用 Tensor Core 进行 FP8 矩阵乘法
    // Hopper Tensor Core 原生支持 FP8
    
    // 声明 Fragment
    wmma::fragment<wmma::matrix_a, 16, 16, 32, __nv_fp8_e4m3, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 32, __nv_fp8_e4m3, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 32, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    for (int k = 0; k < K; k += 32) {
        wmma::load_matrix_sync(a_frag, A + row * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
}
```

### FP8 量化/反量化

```cpp
// FP32 → FP8 量化
__device__ __nv_fp8_e4m3 quantize_fp8(float val, float scale) {
    return __nv_fp8_e4m3(val / scale);
}

// FP8 → FP32 反量化
__device__ float dequantize_fp8(__nv_fp8_e4m3 val, float scale) {
    return float(val) * scale;
}

// 动态缩放
__global__ void compute_scale_kernel(const float* input, float* scale, int n) {
    // 找到最大绝对值
    float max_abs = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        max_abs = fmaxf(max_abs, fabsf(input[i]));
    }
    max_abs = block_reduce_max(max_abs);
    
    if (threadIdx.x == 0) {
        // E4M3 最大值约为 240
        *scale = max_abs / 240.0f;
    }
}
```

---

## 5. Asynchronous Transaction Barrier

### 什么是 Transaction Barrier？

Transaction Barrier 是一种新的同步原语，可以等待特定数量的异步操作完成。

```cpp
#include <cuda/barrier>

__global__ void async_barrier_kernel(float* data) {
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    __shared__ float smem[256];
    
    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x);
    }
    __syncthreads();
    
    // 异步加载
    cuda::memcpy_async(
        smem + threadIdx.x,
        data + blockIdx.x * blockDim.x + threadIdx.x,
        sizeof(float),
        barrier
    );
    
    // 等待所有异步操作完成
    barrier.arrive_and_wait();
    
    // 现在可以安全使用 smem
    float val = smem[threadIdx.x];
}
```

### 多阶段流水线

```cpp
__global__ void pipelined_kernel(const float* input, float* output, int n) {
    constexpr int STAGES = 3;
    __shared__ float smem[STAGES][256];
    __shared__ cuda::barrier<cuda::thread_scope_block> barriers[STAGES];
    
    // 初始化 barriers
    if (threadIdx.x == 0) {
        for (int i = 0; i < STAGES; ++i) {
            init(&barriers[i], blockDim.x);
        }
    }
    __syncthreads();
    
    // Prologue: 填充流水线
    for (int stage = 0; stage < STAGES - 1; ++stage) {
        int idx = stage * blockDim.x + threadIdx.x;
        if (idx < n) {
            cuda::memcpy_async(smem[stage] + threadIdx.x, input + idx, sizeof(float), barriers[stage]);
        }
    }
    
    // Main loop
    for (int i = 0; i < n / blockDim.x; ++i) {
        int compute_stage = i % STAGES;
        int load_stage = (i + STAGES - 1) % STAGES;
        
        // 等待当前阶段数据就绪
        barriers[compute_stage].arrive_and_wait();
        
        // 计算
        float val = smem[compute_stage][threadIdx.x] * 2.0f;
        
        // 异步加载下一批数据
        int next_idx = (i + STAGES - 1) * blockDim.x + threadIdx.x;
        if (next_idx < n) {
            cuda::memcpy_async(smem[load_stage] + threadIdx.x, input + next_idx, sizeof(float), barriers[load_stage]);
        }
        
        // 写回结果
        output[i * blockDim.x + threadIdx.x] = val;
    }
}
```

---

## 6. 性能优化建议

### 何时使用 TMA

- 大块连续数据搬运
- 多维 Tensor 访问
- 需要解放 SM 计算资源

### 何时使用 Cluster

- Block 间需要通信
- 归约操作跨多个 Block
- 需要更大的 Shared Memory

### 何时使用 FP8

- 推理场景
- 模型精度允许
- 需要最大吞吐量

---

## 7. 兼容性注意事项

```cpp
// 检查 Hopper 特性支持
int device;
cudaGetDevice(&device);

cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device);

// Compute Capability 9.0+ 支持 Hopper 特性
if (prop.major >= 9) {
    // 可以使用 TMA, Cluster, FP8
}

// 检查 Cluster 支持
int cluster_support;
cudaDeviceGetAttribute(&cluster_support, cudaDevAttrClusterLaunch, device);
```

---

## 8. 参考资料

- [CUDA C++ Programming Guide - Hopper](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hopper-architecture)
- [NVIDIA H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [CUTLASS 3.0 - Hopper Support](https://github.com/NVIDIA/cutlass/blob/main/media/docs/hopper.md)
