---
outline: [2, 3]
---

# 代码风格指南

本文档定义 CUDA Kernel Academy 项目的代码风格和技术标准。

## C++/CUDA 标准

| 模块 | 最低标准 | 推荐标准 |
|------|----------|----------|
| 01-sgemm-tutorial | C++17 | C++17 |
| 02-tensorcraft-core | C++17 | C++20 |
| 03-hpc-advanced | C++20 | C++20 |
| 04-inference-engine | C++17 | C++17 |

```cmake
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
```

## 命名规范

### 文件命名

```text
cuda_check.hpp    # C++ 头文件 (.hpp)
gemm.cuh          # CUDA 头文件 (.cuh)
inference_engine.cpp  # C++ 源文件 (.cpp)
tiled_gemm.cu     # CUDA 源文件 (.cu)
test_gemm.cpp     # 测试文件 (test_ 前缀)
```

### 代码命名

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

// 私有成员: trailing underscore
class MyClass {
private:
    int value_;
    float* data_;
};
```

### 模块命名空间

```cpp
namespace sgemm { }           // 01-sgemm-tutorial
namespace tensorcraft { }     // 02-tensorcraft-core
namespace hpc { }             // 03-hpc-advanced
namespace mini_inference { }   // 04-inference-engine
```

## CUDA 错误处理

使用 `common/include/cuda_academy/` 中的共享宏：

```cpp
#include "cuda_academy/cuda_academy.hpp"

CUDA_CHECK(cudaMalloc(&ptr, size));
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

## CMake 规范

```cmake
# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA 标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 架构 (Volta 到 Hopper)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures")

# 编译选项
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
)
```

## 代码格式化

项目使用 `.clang-format` 和 `.editorconfig` 定义格式化规则。

```bash
# 格式化单个文件
clang-format -i src/gemm.cu

# 运行 pre-commit 检查
pre-commit run --all-files
```

## 提交前检查清单

- [ ] 代码通过编译，无警告
- [ ] 遵循命名规范
- [ ] 使用 RAII 管理资源
- [ ] 添加 CUDA 错误检查
- [ ] 运行 `pre-commit run --all-files`
- [ ] 添加单元测试

## References
