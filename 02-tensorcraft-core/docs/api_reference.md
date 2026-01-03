# TensorCraft-HPC API Reference

本文档提供 TensorCraft-HPC 库的完整 API 参考。

## 目录

- [Core 模块](#core-模块)
- [Memory 模块](#memory-模块)
- [Kernels 模块](#kernels-模块)
  - [Elementwise](#elementwise)
  - [Softmax](#softmax)
  - [Normalization](#normalization)
  - [GEMM](#gemm)
  - [Attention](#attention)
  - [Conv2D](#conv2d)
  - [Sparse](#sparse)
  - [Fusion](#fusion)

---

## Core 模块

### cuda_check.hpp

CUDA 错误检查宏和工具。

```cpp
#include "tensorcraft/core/cuda_check.hpp"

// 宏定义
TC_CUDA_CHECK(err)      // 检查 CUDA 错误，失败时抛出异常
TC_CUDA_CHECK_LAST()    // 检查最后一个 CUDA 错误
```

### features.hpp

编译时特性检测。

```cpp
#include "tensorcraft/core/features.hpp"

// 预定义宏
TC_CPP17              // C++17 可用
TC_CPP20              // C++20 可用
TC_CPP23              // C++23 可用
TC_CUDA_VERSION       // CUDA 版本号
TC_HAS_WMMA           // 支持 WMMA (Tensor Core)
TC_HAS_FP16           // 支持 FP16
TC_HAS_BF16           // 支持 BF16
TC_HAS_FP8            // 支持 FP8 (CUDA 12.0+)
```

### type_traits.hpp

类型特征和 Concepts。

```cpp
#include "tensorcraft/core/type_traits.hpp"

namespace tensorcraft {
// 类型检测
template<typename T> inline constexpr bool is_half_v;
template<typename T> inline constexpr bool is_bfloat16_v;
template<typename T> inline constexpr bool is_numeric_v;
template<typename T> inline constexpr bool is_floating_point_v;

// C++20 Concepts (如果可用)
template<typename T> concept Numeric;
template<typename T> concept FloatingPoint;
}
```

---

## Memory 模块

### aligned_vector.hpp

对齐向量类型，用于向量化内存访问。

```cpp
#include "tensorcraft/memory/aligned_vector.hpp"

namespace tensorcraft::memory {

template<typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector {
    T val[N];
    __device__ __host__ T& operator[](int i);
    __device__ __host__ const T& operator[](int i) const;
};

// 常用类型别名
using float4_aligned = AlignedVector<float, 4>;
using half8_aligned = AlignedVector<__half, 8>;
}
```

### tensor.hpp

RAII 风格的 Tensor 封装。

```cpp
#include "tensorcraft/memory/tensor.hpp"

namespace tensorcraft::memory {

template<typename T>
class Tensor {
public:
    // 构造函数
    explicit Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, const T* host_data);
    
    // 移动语义
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    // 访问器
    T* data();
    const T* data() const;
    size_t size() const;
    const std::vector<size_t>& shape() const;
    size_t ndim() const;
    
    // 数据传输
    void copy_from_host(const T* host_data);
    void copy_to_host(T* host_data) const;
    std::vector<T> to_vector() const;
    
    // 填充
    void fill(T value);
    void zero();
};
}
```

### memory_pool.hpp

CUDA 内存池管理。

```cpp
#include "tensorcraft/memory/memory_pool.hpp"

namespace tensorcraft::memory {

class MemoryPool {
public:
    static MemoryPool& instance();
    
    void* allocate(size_t bytes);
    void deallocate(void* ptr);
    void release_all();
    
    size_t allocated_bytes() const;
    size_t cached_bytes() const;
};
}
```

---

## Kernels 模块

### Elementwise

```cpp
#include "tensorcraft/kernels/elementwise.hpp"

namespace tensorcraft::kernels {

// 激活函数
template<typename T>
void relu(const T* input, T* output, size_t n, cudaStream_t stream = 0);

template<typename T>
void gelu(const T* input, T* output, size_t n, cudaStream_t stream = 0);

template<typename T>
void silu(const T* input, T* output, size_t n, cudaStream_t stream = 0);

template<typename T>
void sigmoid(const T* input, T* output, size_t n, cudaStream_t stream = 0);

template<typename T>
void tanh_activation(const T* input, T* output, size_t n, cudaStream_t stream = 0);

// 向量运算
template<typename T>
void vector_add(const T* a, const T* b, T* c, size_t n, cudaStream_t stream = 0);

template<typename T>
void vector_mul(const T* a, const T* b, T* c, size_t n, cudaStream_t stream = 0);

template<typename T>
void vector_scale(const T* input, T* output, T scale, size_t n, cudaStream_t stream = 0);

// 通用 Elementwise 启动器
template<typename T, typename Func>
void launch_elementwise(const T* input, T* output, size_t n, Func func, 
                        cudaStream_t stream = 0);

// 预定义 Functors
struct ReLU;
struct GeLU;
struct SiLU;
struct Sigmoid;
struct Tanh;
template<typename T> struct LeakyReLU { T alpha; };
}
```

### Softmax

```cpp
#include "tensorcraft/kernels/softmax.hpp"

namespace tensorcraft::kernels {

// Softmax
template<typename T>
void softmax(const T* input, T* output, int batch_size, int dim, 
             cudaStream_t stream = 0);

// Log Softmax
template<typename T>
void log_softmax(const T* input, T* output, int batch_size, int dim,
                 cudaStream_t stream = 0);

// 带温度的 Softmax
template<typename T>
void softmax_with_temperature(const T* input, T* output, int batch_size, 
                              int dim, float temperature, cudaStream_t stream = 0);
}
```

### Normalization

```cpp
#include "tensorcraft/kernels/normalization.hpp"

namespace tensorcraft::kernels {

// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
template<typename T>
void layernorm(const T* input, const T* gamma, const T* beta, T* output,
               int batch_size, int hidden_size, float eps = 1e-5f,
               cudaStream_t stream = 0);

// RMSNorm: y = x / RMS(x) * weight
template<typename T>
void rmsnorm(const T* input, const T* weight, T* output,
             int batch_size, int hidden_size, float eps = 1e-5f,
             cudaStream_t stream = 0);

// BatchNorm (推理模式)
template<typename T>
void launch_batchnorm(const T* input, const T* gamma, const T* beta,
                      const T* running_mean, const T* running_var, T* output,
                      int N, int C, int H, int W, float eps = 1e-5f,
                      bool fuse_relu = false, cudaStream_t stream = 0);
}
```

### GEMM

```cpp
#include "tensorcraft/kernels/gemm.hpp"

namespace tensorcraft::kernels {

// GEMM 版本枚举
enum class GemmVersion {
    Naive,        // 朴素实现
    Tiled,        // 共享内存分块
    DoubleBuffer, // 双缓冲
    TensorCore    // WMMA Tensor Core
};

// 通用 GEMM: C = alpha * A * B + beta * C
template<typename T>
void gemm(const T* A, const T* B, T* C, int M, int N, int K,
          float alpha = 1.0f, float beta = 0.0f, cudaStream_t stream = 0);

// 指定版本的 GEMM
template<typename T>
void launch_gemm(const T* A, const T* B, T* C, int M, int N, int K,
                 float alpha, float beta, GemmVersion version,
                 cudaStream_t stream = 0);

// WMMA Tensor Core GEMM (half -> float)
void launch_gemm_wmma(const __half* A, const __half* B, float* C,
                      int M, int N, int K, cudaStream_t stream = 0);

// 矩阵转置
template<typename T>
void transpose(const T* input, T* output, int rows, int cols,
               cudaStream_t stream = 0);

// 批量 GEMM
template<typename T>
void batched_gemm(const T* const* A, const T* const* B, T** C,
                  int M, int N, int K, int batch_size,
                  float alpha = 1.0f, float beta = 0.0f,
                  cudaStream_t stream = 0);
}
```

### Attention

```cpp
#include "tensorcraft/kernels/attention.hpp"

namespace tensorcraft::kernels {

// FlashAttention 风格的注意力计算
template<typename T>
void launch_flash_attention(const T* Q, const T* K, const T* V, T* O,
                            int batch_size, int num_heads, int seq_len,
                            int head_dim, float scale,
                            const T* mask = nullptr,
                            cudaStream_t stream = 0);

// 标准多头注意力
template<typename T>
void launch_multihead_attention(const T* Q, const T* K, const T* V, T* O,
                                int batch_size, int num_heads, int seq_len,
                                int head_dim, float scale,
                                cudaStream_t stream = 0);

// RoPE 位置编码
template<typename T>
void precompute_rope_cache(T* cos_cache, T* sin_cache,
                           int max_seq_len, int head_dim,
                           float base = 10000.0f,
                           cudaStream_t stream = 0);

template<typename T>
void launch_rope(T* x, const T* cos_cache, const T* sin_cache,
                 int batch_size, int seq_len, int num_heads, int head_dim,
                 int start_pos = 0, cudaStream_t stream = 0);

// PagedAttention (用于 KV Cache)
template<typename T>
void launch_paged_attention(const T* Q, const T* K_cache, const T* V_cache,
                            T* O, const int* block_tables,
                            const int* context_lens,
                            int batch_size, int num_heads, int head_dim,
                            int block_size, int max_blocks,
                            float scale, cudaStream_t stream = 0);

// MoE 路由
template<typename T>
void launch_moe_router(const T* gate_logits, int* expert_indices,
                       T* expert_weights, int batch_size,
                       int num_experts, int top_k,
                       cudaStream_t stream = 0);
}
```

### Conv2D

```cpp
#include "tensorcraft/kernels/conv2d.hpp"

namespace tensorcraft::kernels {

// 标准 Conv2D
template<typename T>
void conv2d(const T* input, const T* weight, const T* bias, T* output,
            int N, int C, int H, int W, int K, int R, int S,
            int stride = 1, int padding = 0, cudaStream_t stream = 0);

// Depthwise Conv2D
template<typename T>
void conv2d_depthwise(const T* input, const T* weight, const T* bias, T* output,
                      int N, int C, int H, int W, int R, int S,
                      int stride = 1, int padding = 0,
                      cudaStream_t stream = 0);

// Pointwise Conv2D (1x1 卷积)
template<typename T>
void conv2d_pointwise(const T* input, const T* weight, const T* bias, T* output,
                      int N, int C, int H, int W, int K,
                      cudaStream_t stream = 0);

// Im2Col 变换
template<typename T>
void launch_im2col(const T* input, T* col,
                   int N, int C, int H, int W, int R, int S,
                   int stride_h, int stride_w, int pad_h, int pad_w,
                   cudaStream_t stream = 0);

// Col2Im 变换 (反向传播用)
template<typename T>
void launch_col2im(const T* col, T* input,
                   int N, int C, int H, int W, int R, int S,
                   int stride_h, int stride_w, int pad_h, int pad_w,
                   cudaStream_t stream = 0);
}
```

### Sparse

```cpp
#include "tensorcraft/kernels/sparse.hpp"

namespace tensorcraft::kernels {

// CSR 格式的 SpMV: y = A * x
template<typename T>
void launch_spmv_csr(const T* values, const int* col_indices,
                     const int* row_ptrs, const T* x, T* y,
                     int rows, cudaStream_t stream = 0);

// CSC 格式的 SpMV
template<typename T>
void launch_spmv_csc(const T* values, const int* row_indices,
                     const int* col_ptrs, const T* x, T* y,
                     int rows, int cols, cudaStream_t stream = 0);

// CSR 格式的 SpMM: C = A * B
template<typename T>
void launch_spmm_csr(const T* A_values, const int* A_col_indices,
                     const int* A_row_ptrs, const T* B, T* C,
                     int M, int K, int N, cudaStream_t stream = 0);

// 稀疏矩阵格式转换
template<typename T>
void csr_to_csc(const T* csr_values, const int* csr_col_indices,
                const int* csr_row_ptrs, T* csc_values,
                int* csc_row_indices, int* csc_col_ptrs,
                int rows, int cols, int nnz, cudaStream_t stream = 0);
}
```

### Fusion

```cpp
#include "tensorcraft/kernels/fusion.hpp"

namespace tensorcraft::kernels {

// GEMM + Bias + Activation 融合
template<typename T>
void gemm_bias_relu(const T* A, const T* B, const T* bias, T* C,
                    int M, int N, int K, cudaStream_t stream = 0);

template<typename T>
void gemm_bias_gelu(const T* A, const T* B, const T* bias, T* C,
                    int M, int N, int K, cudaStream_t stream = 0);

// Epilogue Functors
struct EpilogueIdentity;
struct EpilogueBias;
struct EpilogueBiasReLU;
struct EpilogueBiasGeLU;

// 通用融合 GEMM
template<typename T, typename Epilogue>
void launch_gemm_fused(const T* A, const T* B, T* C, int M, int N, int K,
                       Epilogue epilogue, cudaStream_t stream = 0);

// INT8 量化
template<typename T>
void quantize_int8(const T* input, int8_t* output, float scale,
                   int8_t zero_point, size_t n, cudaStream_t stream = 0);

template<typename T>
void dequantize_int8(const int8_t* input, T* output, float scale,
                     int8_t zero_point, size_t n, cudaStream_t stream = 0);

// FP8 量化 (CUDA 12.0+)
#ifdef TC_HAS_FP8
template<typename T>
void quantize_fp8(const T* input, __nv_fp8_e4m3* output, float scale,
                  size_t n, cudaStream_t stream = 0);

template<typename T>
void dequantize_fp8(const __nv_fp8_e4m3* input, T* output, float scale,
                    size_t n, cudaStream_t stream = 0);
#endif
}
```

---

## Python API

```python
import tensorcraft_ops as tc

# Elementwise
tc.relu(input: np.ndarray) -> np.ndarray
tc.gelu(input: np.ndarray) -> np.ndarray
tc.silu(input: np.ndarray) -> np.ndarray
tc.sigmoid(input: np.ndarray) -> np.ndarray

# Softmax
tc.softmax(input: np.ndarray, dim: int = -1) -> np.ndarray

# Normalization
tc.layernorm(input: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
             eps: float = 1e-5) -> np.ndarray
tc.rmsnorm(input: np.ndarray, weight: np.ndarray, 
           eps: float = 1e-5) -> np.ndarray

# GEMM
tc.gemm(A: np.ndarray, B: np.ndarray, 
        version: str = 'tiled') -> np.ndarray
# version: 'naive', 'tiled', 'double_buffer'

# Transpose
tc.transpose(input: np.ndarray) -> np.ndarray
```

---

## 错误处理

所有 CUDA 操作都会检查错误。失败时抛出 `std::runtime_error`：

```cpp
try {
    tensorcraft::kernels::gemm(A, B, C, M, N, K);
} catch (const std::runtime_error& e) {
    std::cerr << "CUDA error: " << e.what() << std::endl;
}
```

## 流支持

所有 kernel 函数都支持可选的 CUDA stream 参数：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

tensorcraft::kernels::gemm(A, B, C, M, N, K, 1.0f, 0.0f, stream);

cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
```
