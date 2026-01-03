# 架构设计文档

## 概述

Mini-Inference Engine 是一个轻量级的神经网络推理引擎，专注于 GEMM（通用矩阵乘法）优化。本文档详细描述系统的架构设计。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Application Layer                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Benchmark  │  │ MNIST Demo  │  │   Tests     │  │  User App   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Engine Layer                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      InferenceEngine                                 │   │
│  │  - Weight Loading    - Forward Pass    - Layer Management           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Tensor    │  │  AutoTuner  │  │  Profiler   │  │   Config    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Kernel Layer                                       │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐     │
│  │   Naive   │ │   Tiled   │ │ Coalesced │ │  Double   │ │ Optimized │     │
│  │  MatMul   │ │   GEMM    │ │   GEMM    │ │  Buffer   │ │   GEMM    │     │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘     │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐                   │
│  │   Fused   │ │Vectorized │ │   Half    │ │  Batched  │                   │
│  │   GEMM    │ │   GEMM    │ │   GEMM    │ │   GEMM    │                   │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Infrastructure Layer                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │MemoryPool  │  │StreamManager│  │   Logger    │  │Quantization │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│  ┌─────────────┐  ┌─────────────┐                                          │
│  │DeviceMemory│  │  GpuTimer   │                                          │
│  └─────────────┘  └─────────────┘                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CUDA Runtime                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         │
│  │   cudart    │  │   cuBLAS    │  │   Streams   │                         │
│  └─────────────┘  └─────────────┘  └─────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. InferenceEngine

推理引擎的核心类，负责管理神经网络的前向传播。

```cpp
class InferenceEngine {
    // 生命周期管理
    void init(int device_id);
    void cleanup();
    
    // 权重管理
    bool load_weights(const std::string& path);
    bool save_weights(const std::string& path);
    
    // 推理
    void forward(const float* input, float* output, int batch_size);
    void forward_with_timing(const float* input, float* output, 
                            int batch_size, std::vector<float>& layer_times);
};
```

**设计决策**:
- 使用 RAII 模式管理 GPU 资源
- 支持多层网络的链式前向传播
- 使用融合 kernel 减少内存带宽消耗

### 2. Tensor

N 维张量类，提供 GPU 存储和基本操作。

```cpp
class Tensor {
    // 存储
    std::vector<int> shape_;
    std::vector<int> strides_;
    PooledMemory data_;
    
    // 操作
    void fill(float value);
    void zero();
    Tensor clone() const;
    void reshape(const std::vector<int>& new_shape);
};
```

**设计决策**:
- 使用内存池减少分配开销
- 禁用拷贝构造，强制使用移动语义
- 支持任意维度的张量

### 3. MemoryPool

GPU 内存池，通过缓存减少 cudaMalloc 调用。

```cpp
class MemoryPool {
    // 分配策略
    void* allocate(size_t size);      // 优先从缓存分配
    void deallocate(void* ptr);       // 返回缓存而非释放
    
    // 缓存管理
    void clear_cache();               // 释放缓存
    void clear_all();                 // 释放所有内存
};
```

**设计决策**:
- 使用 best-fit 策略匹配缓存块
- 256 字节对齐以支持向量化加载
- 线程安全设计

### 4. StreamManager

CUDA 流管理器，支持并发执行。

```cpp
class StreamManager {
    // 流管理
    void init(int num_streams);
    cudaStream_t get_stream();        // 轮询分配
    cudaStream_t get_stream(int idx); // 指定索引
    
    // 同步
    void sync_all();
    void sync(int index);
};
```

**设计决策**:
- 单例模式确保全局一致性
- 轮询分配实现负载均衡
- 支持细粒度同步

## GEMM Kernel 设计

### 优化层次

```
Level 1: Naive
    │
    ▼ 共享内存
Level 2: Tiled
    │
    ▼ 内存合并
Level 3: Coalesced
    │
    ▼ 延迟隐藏
Level 4: Double Buffer
    │
    ▼ 寄存器分块
Level 5: Register Blocked
    │
    ▼ 算子融合
Level 6: Fused
    │
    ▼ 向量化
Level 7: Vectorized
```

### Kernel 参数设计

```cpp
// 模板参数
template<
    int BM,    // Block 处理的 M 维度
    int BN,    // Block 处理的 N 维度
    int BK,    // 每次迭代的 K 维度
    int TM,    // 每线程处理的 M 维度
    int TN     // 每线程处理的 N 维度
>
__global__ void optimized_gemm(...);
```

**参数约束**:
```
threads_per_block = (BM / TM) * (BN / TN) <= 1024
shared_memory = (BM * BK + BK * BN) * sizeof(float) <= 48KB
registers_per_thread = TM * TN + TM + TN + overhead <= 255
```

### 内存访问模式

```
Global Memory Layout (Row-Major):
┌─────────────────────────────────────┐
│ A[0,0] A[0,1] A[0,2] ... A[0,K-1]  │
│ A[1,0] A[1,1] A[1,2] ... A[1,K-1]  │
│ ...                                 │
│ A[M-1,0] ...           A[M-1,K-1]  │
└─────────────────────────────────────┘

Shared Memory Tiling:
┌─────────┐     ┌─────────┐
│ A_tile  │  ×  │ B_tile  │  =  C_tile
│ BM × BK │     │ BK × BN │     BM × BN
└─────────┘     └─────────┘
```

## 数据流

### 推理数据流

```
Input (Host)
    │
    ▼ cudaMemcpy H2D
Input (Device)
    │
    ▼ Layer 0: GEMM + Bias + ReLU
Hidden 0 (Device)
    │
    ▼ Layer 1: GEMM + Bias + ReLU
Hidden 1 (Device)
    │
    ▼ Layer N: GEMM + Bias
Output (Device)
    │
    ▼ cudaMemcpy D2H
Output (Host)
```

### 权重加载流程

```
Weight File (.bin)
    │
    ▼ fread()
Host Buffer
    │
    ▼ cudaMemcpy H2D
Device Weights
    │
    ▼ 存储到 InferenceEngine
Ready for Inference
```

## 错误处理

### CUDA 错误处理

```cpp
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw CudaException(err, __FILE__, __LINE__); \
    } \
} while(0)
```

### 异常层次

```
std::exception
    │
    ├── CudaException      // CUDA 运行时错误
    │
    ├── std::runtime_error // cuBLAS 错误
    │
    └── std::invalid_argument // 参数验证错误
```

## 线程安全

### 线程安全组件

| 组件 | 线程安全 | 机制 |
|------|----------|------|
| MemoryPool | ✓ | std::mutex |
| StreamManager | ✓ | std::mutex |
| Logger | ✓ | std::mutex |
| Config | ✓ | 只读访问 |
| Tensor | ✗ | 单线程使用 |
| InferenceEngine | ✗ | 单线程使用 |

### 并发模式

```cpp
// 推荐: 每线程一个 InferenceEngine
void worker_thread(int thread_id) {
    InferenceEngine engine;
    engine.init(thread_id % num_gpus);
    engine.load_weights("model.bin");
    
    while (has_work()) {
        engine.forward(input, output, batch_size);
    }
    
    engine.cleanup();
}
```

## 扩展点

### 添加新 Kernel

1. 在 `include/kernels.cuh` 声明
2. 在 `src/` 实现
3. 在 `GemmKernelType` 添加枚举
4. 在 `AutoTuner` 注册

### 添加新激活函数

1. 在 `fused_gemm.cu` 添加模板特化
2. 在 `FusionConfig` 添加配置
3. 更新 `launch_fused_gemm` 分发逻辑

### 添加新数据类型

1. 参考 `half_gemm.cuh` 实现
2. 添加类型转换函数
3. 更新 Tensor 类支持多类型
