# CUDA Kernel Academy 代码风格指南

本文档定义了 CUDA Kernel Academy 项目的统一代码风格和技术标准。

## 目录

1. [C++ 标准](#c-标准)
2. [命名规范](#命名规范)
3. [文件组织](#文件组织)
4. [CUDA 编程规范](#cuda-编程规范)
5. [错误处理](#错误处理)
6. [内存管理](#内存管理)
7. [文档注释](#文档注释)
8. [CMake 规范](#cmake-规范)

---

## C++ 标准

### 版本要求

| 模块 | 最低标准 | 推荐标准 | 说明 |
|------|----------|----------|------|
| 01-sgemm-tutorial | C++17 | C++17 | 入门教程，保持简单 |
| 02-tensorcraft-core | C++17 | C++20 | 核心库，支持更高版本 |
| 03-hpc-advanced | C++20 | C++20 | 进阶特性，使用现代 C++ |
| 04-inference-engine | C++17 | C++17 | 应用框架，兼容性优先 |

### 推荐的 C++ 特性

```cpp
// ✅ 推荐使用
auto                          // 类型推导
constexpr                     // 编译期计算
[[nodiscard]]                 // 返回值检查
std::optional<T>              // 可选值
std::string_view              // 字符串视图
std::span<T>                  // 数组视图 (C++20)
if constexpr                  // 编译期分支
structured bindings           // auto [a, b] = ...
std::unique_ptr / std::shared_ptr  // 智能指针

// ❌ 避免使用
raw new/delete                // 使用智能指针
C-style casts                 // 使用 static_cast 等
#define for constants         // 使用 constexpr
```

### CUDA 标准

```cmake
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
```

---

## 命名规范

### 文件命名

```
// 头文件
cuda_check.hpp          // C++ 头文件 (.hpp)
gemm.cuh                // CUDA 头文件 (.cuh)

// 源文件
inference_engine.cpp    // C++ 源文件 (.cpp)
tiled_gemm.cu           // CUDA 源文件 (.cu)

// 测试文件
test_gemm.cpp           // 测试文件 (test_ 前缀)
```

### 命名风格

```cpp
// 命名空间: snake_case
namespace tensorcraft { }
namespace mini_inference { }
namespace hpc { }

// 类/结构体: PascalCase
class InferenceEngine { };
struct GemmConfig { };

// 函数: snake_case
void launch_gemm(...);
float compute_gflops(...);

// 变量: snake_case
int block_size;
float* device_ptr;

// 常量: kPascalCase 或 UPPER_SNAKE_CASE
constexpr int kBlockSize = 32;
constexpr int TILE_SIZE = 16;

// 宏: UPPER_SNAKE_CASE
#define TC_CUDA_CHECK(call) ...
#define CUDA_CHECK(call) ...

// 模板参数: PascalCase
template <typename T, int BlockSize>

// 枚举: PascalCase (枚举类型), UPPER_SNAKE_CASE (枚举值)
enum class GemmVersion {
    NAIVE,
    TILED,
    TENSOR_CORE
};

// 私有成员: trailing underscore
class MyClass {
private:
    int value_;
    float* data_;
};
```

### 命名空间

```cpp
// 各模块的命名空间
namespace sgemm { }           // 01-sgemm-tutorial
namespace tensorcraft { }     // 02-tensorcraft-core
namespace hpc { }             // 03-hpc-advanced
namespace mini_inference { }  // 04-inference-engine

// 子命名空间
namespace tensorcraft::kernels { }
namespace tensorcraft::memory { }
namespace tensorcraft::core { }
```

---

## 文件组织

### 头文件结构

```cpp
#pragma once
/**
 * @file gemm.hpp
 * @brief GEMM kernel implementations
 * @author CUDA Kernel Academy
 */

// 1. 系统头文件
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>

// 2. 第三方库头文件
#include <cutlass/gemm/device/gemm.h>

// 3. 项目内部头文件
#include "tensorcraft/core/cuda_check.hpp"
#include "tensorcraft/core/type_traits.hpp"

namespace tensorcraft::kernels {

// 4. 前向声明
class GemmConfig;

// 5. 类型定义
using GemmHandle = std::unique_ptr<GemmConfig>;

// 6. 常量定义
constexpr int kDefaultBlockSize = 32;

// 7. 类/结构体定义
class Gemm {
public:
    // ...
};

// 8. 函数声明
void launch_gemm(...);

} // namespace tensorcraft::kernels
```

### 目录结构

```
module/
├── include/                    # 公共头文件
│   └── module_name/
│       ├── core/               # 核心工具
│       ├── kernels/            # 算子实现
│       └── memory/             # 内存管理
├── src/                        # 源文件
│   └── *.cu, *.cpp
├── tests/                      # 测试
│   └── test_*.cpp
├── benchmarks/                 # 性能测试
│   └── *_benchmark.cpp
├── docs/                       # 文档
└── CMakeLists.txt
```

---

## CUDA 编程规范

### Kernel 函数

```cpp
/**
 * @brief Tiled GEMM kernel
 * @tparam BLOCK_M Tile size in M dimension
 * @tparam BLOCK_N Tile size in N dimension
 * @tparam BLOCK_K Tile size in K dimension
 */
template <int BLOCK_M = 32, int BLOCK_N = 32, int BLOCK_K = 8>
__global__ void tiled_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    // 1. 计算线程索引
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // 2. 声明共享内存
    __shared__ float As[BLOCK_M][BLOCK_K];
    __shared__ float Bs[BLOCK_K][BLOCK_N];
    
    // 3. 主计算循环
    float acc = 0.0f;
    
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load tiles to shared memory
        // ...
        __syncthreads();
        
        // Compute partial result
        // ...
        __syncthreads();
    }
    
    // 4. 写回结果
    // ...
}
```

### Launch 函数

```cpp
/**
 * @brief Launch GEMM kernel
 * @param A Input matrix A [M x K]
 * @param B Input matrix B [K x N]
 * @param C Output matrix C [M x N]
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha Scalar multiplier for A*B
 * @param beta Scalar multiplier for C
 * @param version GEMM implementation version
 * @param stream CUDA stream (optional)
 */
void launch_gemm(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    GemmVersion version = GemmVersion::TILED,
    cudaStream_t stream = nullptr
);
```

### 常量和配置

```cpp
// 使用 constexpr 定义编译期常量
namespace config {
    constexpr int kWarpSize = 32;
    constexpr int kMaxBlockSize = 1024;
    constexpr int kDefaultTileM = 128;
    constexpr int kDefaultTileN = 128;
    constexpr int kDefaultTileK = 8;
}

// 使用结构体组织配置
struct GemmConfig {
    int block_m = config::kDefaultTileM;
    int block_n = config::kDefaultTileN;
    int block_k = config::kDefaultTileK;
    bool use_tensor_core = false;
    cudaStream_t stream = nullptr;
};
```

---

## 错误处理

### 统一的错误检查宏

```cpp
// 推荐的错误检查风格 (来自 tensorcraft-core)
#define TC_CUDA_CHECK(call)                                     \
    do {                                                        \
        cudaError_t err = (call);                               \
        if (err != cudaSuccess) {                               \
            throw ::tensorcraft::CudaException(__FILE__, __LINE__, err); \
        }                                                       \
    } while (0)

#define TC_CUDA_CHECK_LAST()                                    \
    do {                                                        \
        cudaError_t err = cudaGetLastError();                   \
        if (err != cudaSuccess) {                               \
            throw ::tensorcraft::CudaException(__FILE__, __LINE__, err); \
        }                                                       \
    } while (0)

#define TC_CUDA_SYNC_CHECK()                                    \
    do {                                                        \
        TC_CUDA_CHECK(cudaDeviceSynchronize());                 \
        TC_CUDA_CHECK_LAST();                                   \
    } while (0)
```

### 异常类

```cpp
class CudaException : public std::runtime_error {
public:
    CudaException(const std::string& file, int line, cudaError_t error)
        : std::runtime_error(format_message(file, line, error))
        , error_(error)
        , file_(file)
        , line_(line) {}
    
    cudaError_t error() const noexcept { return error_; }
    const std::string& file() const noexcept { return file_; }
    int line() const noexcept { return line_; }

private:
    cudaError_t error_;
    std::string file_;
    int line_;
};
```

---

## 内存管理

### RAII 包装器

```cpp
template <typename T>
class DeviceMemory {
public:
    DeviceMemory() = default;
    
    explicit DeviceMemory(size_t count) : size_(count) {
        TC_CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }
    
    ~DeviceMemory() {
        if (ptr_) cudaFree(ptr_);
    }
    
    // Move semantics
    DeviceMemory(DeviceMemory&& other) noexcept;
    DeviceMemory& operator=(DeviceMemory&& other) noexcept;
    
    // Disable copy
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    
    // Accessors
    T* get() noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }
    size_t size() const noexcept { return size_; }
    
    // Operations
    void copy_from_host(const T* host_ptr, size_t count);
    void copy_to_host(T* host_ptr, size_t count) const;
    void zero();

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};
```

### 智能指针使用

```cpp
// 使用 unique_ptr 管理资源
auto engine = std::make_unique<InferenceEngine>();

// 使用 shared_ptr 共享资源
auto config = std::make_shared<GemmConfig>();

// 自定义删除器
auto cuda_deleter = [](float* ptr) { cudaFree(ptr); };
std::unique_ptr<float, decltype(cuda_deleter)> device_ptr(nullptr, cuda_deleter);
```

---

## 文档注释

### Doxygen 风格

```cpp
/**
 * @brief Compute matrix multiplication C = alpha * A * B + beta * C
 * 
 * This function performs a general matrix multiplication (GEMM) operation
 * using optimized CUDA kernels.
 * 
 * @tparam T Data type (float, half, etc.)
 * 
 * @param[in] A Input matrix A of size [M x K]
 * @param[in] B Input matrix B of size [K x N]
 * @param[in,out] C Output matrix C of size [M x N]
 * @param[in] M Number of rows in A and C
 * @param[in] N Number of columns in B and C
 * @param[in] K Number of columns in A and rows in B
 * @param[in] alpha Scalar multiplier for A*B (default: 1.0)
 * @param[in] beta Scalar multiplier for C (default: 0.0)
 * @param[in] stream CUDA stream for async execution (default: nullptr)
 * 
 * @return void
 * 
 * @throws CudaException if CUDA operation fails
 * @throws std::invalid_argument if dimensions are invalid
 * 
 * @note All matrices must be in row-major format
 * @note Pointers must be device pointers
 * 
 * @example
 * @code
 * float *d_A, *d_B, *d_C;
 * // ... allocate and initialize ...
 * launch_gemm(d_A, d_B, d_C, 1024, 1024, 1024);
 * @endcode
 * 
 * @see GemmVersion for available implementations
 */
template <typename T>
void launch_gemm(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    T alpha = T(1), T beta = T(0),
    cudaStream_t stream = nullptr
);
```

---

## CMake 规范

### 统一的 CMake 配置

```cmake
cmake_minimum_required(VERSION 3.20)
project(ModuleName LANGUAGES CXX CUDA)

# ============================================================================
# C++ 标准配置
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# CUDA 配置
# ============================================================================
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 架构 (Volta 到 Hopper)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures")

# ============================================================================
# 编译选项
# ============================================================================
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --extended-lambda
        -Xcompiler=-Wall
    >
)

# Release 优化
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG")

# ============================================================================
# 依赖管理
# ============================================================================
include(FetchContent)

# GoogleTest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
```

---

## 代码格式化

### clang-format 配置

在项目根目录创建 `.clang-format`:

```yaml
BasedOnStyle: Google
IndentWidth: 4
ColumnLimit: 100
PointerAlignment: Left
ReferenceAlignment: Left
SpaceAfterCStyleCast: false
SpaceBeforeParens: ControlStatements
AllowShortFunctionsOnASingleLine: Inline
AllowShortIfStatementsOnASingleLine: false
AllowShortLoopsOnASingleLine: false
BreakBeforeBraces: Attach
IndentCaseLabels: true
NamespaceIndentation: None
```

### 使用方法

```bash
# 格式化单个文件
clang-format -i src/gemm.cu

# 格式化整个目录
find src include -name "*.cpp" -o -name "*.cu" -o -name "*.hpp" -o -name "*.cuh" | xargs clang-format -i
```

---

## 检查清单

在提交代码前，请确保：

- [ ] 代码通过编译，无警告
- [ ] 遵循命名规范
- [ ] 使用 RAII 管理资源
- [ ] 添加必要的错误检查
- [ ] 编写 Doxygen 注释
- [ ] 运行 clang-format
- [ ] 添加单元测试
- [ ] 更新相关文档

---
