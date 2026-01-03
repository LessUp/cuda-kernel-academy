# TensorCraft-HPC 架构设计

本文档描述 TensorCraft-HPC 的整体架构和设计决策。

## 设计原则

### 1. Header-Only 设计

核心库采用纯头文件设计，优点：

- **零配置集成**：只需 `#include` 即可使用
- **编译时优化**：模板代码可以完全内联
- **跨平台兼容**：无需预编译库

```cpp
// 使用方式
#include "tensorcraft/kernels/gemm.hpp"
tensorcraft::kernels::gemm(A, B, C, M, N, K);
```

### 2. 渐进式优化

每个算子提供多个优化级别，便于学习和对比：

```
Naive → Tiled → Double Buffer → Tensor Core
  ↓        ↓          ↓              ↓
基础实现  共享内存   隐藏延迟      硬件加速
```

### 3. 现代 C++ 优先

充分利用 C++17/20/23 特性：

- **Concepts** (C++20): 类型约束
- **constexpr if** (C++17): 编译时分支
- **Structured Bindings** (C++17): 多返回值
- **RAII**: 自动资源管理

### 4. 向后兼容

支持 CUDA 11.0-13.1，通过条件编译处理新特性：

```cpp
#if TC_CUDA_VERSION >= 12000
    // FP8 支持
#endif

#if TC_CUDA_VERSION >= 11080
    // WGMMA 支持
#endif
```

---

## 模块架构

```
tensorcraft/
├── core/           # 核心工具层
├── memory/         # 内存管理层
└── kernels/        # 算子实现层
```

### Core 层

提供基础设施：

```
core/
├── cuda_check.hpp    # 错误检查
├── features.hpp      # 特性检测
└── type_traits.hpp   # 类型系统
```

**cuda_check.hpp**: CUDA 错误检查宏

```cpp
#define TC_CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        throw std::runtime_error(cudaGetErrorString(e)); \
    } \
} while(0)
```

**features.hpp**: 编译时特性检测

```cpp
// C++ 版本检测
#if __cplusplus >= 202002L
    #define TC_CPP20 1
#endif

// CUDA 特性检测
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    #define TC_HAS_WMMA 1
#endif
```

**type_traits.hpp**: 类型特征和 Concepts

```cpp
template<typename T>
concept Numeric = std::is_arithmetic_v<T> || is_half_v<T>;
```

### Memory 层

提供内存抽象：

```
memory/
├── aligned_vector.hpp  # 向量化加载
├── tensor.hpp          # Tensor 封装
└── memory_pool.hpp     # 内存池
```

**AlignedVector**: 支持向量化内存访问

```cpp
template<typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector {
    T val[N];
};

// 128-bit 加载
using Vec4 = AlignedVector<float, 4>;
Vec4 data = *reinterpret_cast<const Vec4*>(&input[idx]);
```

**Tensor**: RAII 风格的 GPU 张量

```cpp
template<typename T>
class Tensor {
    T* data_ = nullptr;
    std::vector<size_t> shape_;
    
public:
    Tensor(const std::vector<size_t>& shape);
    ~Tensor() { if (data_) cudaFree(data_); }
    
    // Move-only
    Tensor(Tensor&&) noexcept;
    Tensor(const Tensor&) = delete;
};
```

**MemoryPool**: 减少分配开销

```cpp
class MemoryPool {
    std::map<size_t, std::vector<void*>> free_blocks_;
    
public:
    void* allocate(size_t bytes);
    void deallocate(void* ptr);
};
```

### Kernels 层

算子实现：

```
kernels/
├── elementwise.hpp    # 逐元素操作
├── softmax.hpp        # Softmax
├── normalization.hpp  # 归一化
├── gemm.hpp           # 矩阵乘法
├── attention.hpp      # 注意力机制
├── conv2d.hpp         # 卷积
├── sparse.hpp         # 稀疏矩阵
└── fusion.hpp         # 融合与量化
```

---

## Kernel 设计模式

### 1. Functor 模式

使用 Functor 实现可组合的操作：

```cpp
struct ReLU {
    template<typename T>
    __device__ __forceinline__ T operator()(T x) const {
        return x > T(0) ? x : T(0);
    }
};

template<typename T, typename Func>
__global__ void elementwise_kernel(const T* in, T* out, size_t n, Func func) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = func(in[idx]);
}
```

### 2. 版本选择模式

通过枚举选择优化级别：

```cpp
enum class GemmVersion { Naive, Tiled, DoubleBuffer, TensorCore };

template<typename T>
void launch_gemm(const T* A, const T* B, T* C, int M, int N, int K,
                 GemmVersion version) {
    switch (version) {
        case GemmVersion::Naive:
            gemm_naive<<<grid, block>>>(A, B, C, M, N, K);
            break;
        case GemmVersion::Tiled:
            gemm_tiled<<<grid, block, smem>>>(A, B, C, M, N, K);
            break;
        // ...
    }
}
```

### 3. Epilogue 模式

GEMM 后处理的可扩展设计：

```cpp
struct EpilogueBiasReLU {
    const float* bias;
    
    __device__ __forceinline__ float operator()(float acc, int col) const {
        float result = acc + bias[col];
        return result > 0.0f ? result : 0.0f;
    }
};

template<typename T, typename Epilogue>
__global__ void gemm_with_epilogue(/* ... */, Epilogue epilogue) {
    // ... GEMM 计算 ...
    C[idx] = epilogue(acc, col);
}
```

### 4. 编译时配置

使用模板参数进行编译时配置：

```cpp
template<typename T, int TILE_M = 128, int TILE_N = 128, int TILE_K = 32>
__global__ void gemm_optimized(/* ... */) {
    __shared__ T As[TILE_M][TILE_K];
    __shared__ T Bs[TILE_K][TILE_N];
    // ...
}
```

---

## 内存访问优化

### 向量化加载

```cpp
// 标量加载: 4 次内存事务
float a = input[idx];
float b = input[idx+1];
float c = input[idx+2];
float d = input[idx+3];

// 向量化加载: 1 次内存事务
float4 vec = *reinterpret_cast<const float4*>(&input[idx]);
```

### 合并访问

```cpp
// 好: 相邻线程访问相邻内存
output[threadIdx.x] = input[threadIdx.x];

// 差: 跨步访问
output[threadIdx.x * stride] = input[threadIdx.x * stride];
```

### Bank Conflict 避免

```cpp
// 有 bank conflict
__shared__ float tile[32][32];

// 无 bank conflict (添加 padding)
__shared__ float tile[32][33];
```

---

## 数值稳定性

### Softmax 稳定性

```cpp
// 不稳定: exp 可能溢出
for (int i = 0; i < n; ++i)
    output[i] = exp(input[i]) / sum;

// 稳定: 减去最大值
float max_val = *max_element(input, input + n);
for (int i = 0; i < n; ++i)
    output[i] = exp(input[i] - max_val) / sum;
```

### LayerNorm 稳定性

```cpp
// Welford 算法: 单遍计算均值和方差
float mean = 0.0f, M2 = 0.0f;
for (int i = 0; i < n; ++i) {
    float delta = x[i] - mean;
    mean += delta / (i + 1);
    M2 += delta * (x[i] - mean);
}
float var = M2 / n;
```

---

## 扩展指南

### 添加新算子

1. 在 `include/tensorcraft/kernels/` 创建头文件
2. 实现 kernel 函数和启动器
3. 添加测试到 `tests/`
4. 添加基准测试到 `benchmarks/`
5. 更新文档

### 添加新优化级别

1. 在现有头文件中添加新的 kernel 实现
2. 更新版本枚举
3. 在启动器中添加分支
4. 添加对比测试

### 支持新数据类型

1. 在 `type_traits.hpp` 添加类型检测
2. 在 `features.hpp` 添加特性检测
3. 特化相关 kernel 模板
4. 添加类型转换工具

---

## 性能考量

### 占用率 vs 寄存器

```cpp
// 高占用率: 更多并行
__launch_bounds__(256, 4)  // 256 threads, 4 blocks/SM

// 更多寄存器: 更少溢出
__launch_bounds__(128, 2)  // 128 threads, 2 blocks/SM
```

### 共享内存 vs L1 Cache

```cpp
// 偏向共享内存
cudaFuncSetAttribute(kernel, 
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);

// 偏向 L1 Cache
cudaFuncSetAttribute(kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout, 0);
```

### 异步执行

```cpp
// 使用多个 stream 重叠执行
cudaStream_t streams[4];
for (int i = 0; i < 4; ++i) {
    cudaStreamCreate(&streams[i]);
    kernel<<<grid, block, 0, streams[i]>>>(data[i]);
}
```
