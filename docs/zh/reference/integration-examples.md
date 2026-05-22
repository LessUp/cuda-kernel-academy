---
outline: [2, 3]
---

# 集成示例

本文档展示如何在你的项目中集成和使用 CUDA Kernel Academy 的各个模块。

## 使用 tensorcraft-core 算子库

### 方式一：作为子目录

```cmake
# 你的项目 CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(my_cuda_project LANGUAGES CXX CUDA)

# 添加 tensorcraft-core 作为子目录
add_subdirectory(path/to/02-tensorcraft-core tensorcraft)

# 链接到你的目标
add_executable(my_app main.cu)
target_link_libraries(my_app PRIVATE tensorcraft)
```

### 方式二：使用 FetchContent

```cmake
include(FetchContent)

FetchContent_Declare(
    tensorcraft
    GIT_REPOSITORY https://github.com/AICL-Lab/cuda-kernel-academy.git
    GIT_TAG main
    SOURCE_SUBDIR 02-tensorcraft-core
)

FetchContent_MakeAvailable(tensorcraft)

target_link_libraries(my_app PRIVATE tensorcraft)
```

### 方式三：Header-only 直接包含

由于 tensorcraft-core 是 header-only 库，你也可以直接包含头文件：

```cmake
target_include_directories(my_app PRIVATE
    path/to/02-tensorcraft-core/include
)
```

## CMake 集成方式

### 完整的 CMakeLists.txt 示例

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

set(TENSORCRAFT_DIR "${CMAKE_SOURCE_DIR}/../cuda-kernel-academy/02-tensorcraft-core")
add_subdirectory(${TENSORCRAFT_DIR} tensorcraft EXCLUDE_FROM_ALL)

add_executable(my_app
    src/main.cu
    src/model.cu
)

target_link_libraries(my_app PRIVATE tensorcraft)

target_compile_options(my_app PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        -Xptxas=-v
    >
)
```

## GEMM 调用示例

### 基本 GEMM 调用

```cpp
#include <tensorcraft/kernels/gemm.hpp>
#include <tensorcraft/core/cuda_check.hpp>

int main() {
    const int M = 1024, N = 1024, K = 1024;

    float *d_A, *d_B, *d_C;
    TC_CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    tensorcraft::kernels::launch_gemm(
        d_A, d_B, d_C,
        M, N, K,
        1.0f,  // alpha
        0.0f,  // beta
        tensorcraft::kernels::GemmVersion::TILED
    );

    TC_CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```

### 选择不同的 GEMM 版本

```cpp
using namespace tensorcraft::kernels;

launch_gemm(A, B, C, M, N, K, 1.0f, 0.0f, GemmVersion::NAIVE);
launch_gemm(A, B, C, M, N, K, 1.0f, 0.0f, GemmVersion::TILED);
launch_gemm(A, B, C, M, N, K, 1.0f, 0.0f, GemmVersion::DOUBLE_BUFFER);
launch_gemm(A, B, C, M, N, K, 1.0f, 0.0f, GemmVersion::TENSOR_CORE);
```

## Attention 调用示例

### Flash Attention

```cpp
#include <tensorcraft/kernels/attention.hpp>

void run_attention() {
    const int batch = 4;
    const int seq_len = 512;
    const int num_heads = 8;
    const int head_dim = 64;

    float *d_Q, *d_K, *d_V, *d_O;

    tensorcraft::kernels::launch_flash_attention(
        d_Q, d_K, d_V, d_O,
        batch, seq_len, num_heads, head_dim,
        1.0f / sqrtf(head_dim)
    );
}
```

## 从 04-inference-engine 学习集成模式

`04-inference-engine` 展示了如何将 tensorcraft-core 集成到完整的推理框架中。

### 关键集成点

#### 1. CMakeLists.txt 依赖配置

```cmake
cmake_minimum_required(VERSION 3.20)
project(mini_inference LANGUAGES CXX CUDA)

set(TENSORCRAFT_ROOT "${CMAKE_SOURCE_DIR}/../02-tensorcraft-core")
add_subdirectory(${TENSORCRAFT_ROOT} tensorcraft)

add_library(mini_inference STATIC
    src/inference_engine.cpp
    src/tensor.cu
)

target_link_libraries(mini_inference PUBLIC tensorcraft)
target_include_directories(mini_inference PUBLIC include)
```

#### 2. 代码中使用 tensorcraft

```cpp
#include "inference_engine.h"
#include <tensorcraft/kernels/gemm.hpp>
#include <tensorcraft/kernels/elementwise.hpp>

void InferenceEngine::linear_layer(
    const float* input,
    const float* weight,
    float* output,
    int M, int N, int K
) {
    tensorcraft::kernels::launch_gemm(
        input, weight, output,
        M, N, K,
        1.0f, 0.0f,
        tensorcraft::kernels::GemmVersion::TILED
    );
}

void InferenceEngine::relu_activation(float* data, int size) {
    tensorcraft::kernels::launch_relu(data, size);
}
```

## 最佳实践

### 1. 错误检查

始终使用 tensorcraft 提供的错误检查宏：

```cpp
#include <tensorcraft/core/cuda_check.hpp>

TC_CUDA_CHECK(cudaMalloc(&ptr, size));
TC_CUDA_CHECK_LAST();
TC_CUDA_SYNC_CHECK();
```

### 2. 性能调优

```cpp
GemmVersion select_gemm_version(int M, int N, int K) {
    if (M < 128 || N < 128 || K < 128) {
        return GemmVersion::NAIVE;
    } else if (M >= 1024 && N >= 1024) {
        return GemmVersion::TENSOR_CORE;
    } else {
        return GemmVersion::TILED;
    }
}
```

### 3. 内存管理

结合 tensorcraft 的内存工具：

```cpp
#include <tensorcraft/memory/memory_pool.hpp>

tensorcraft::memory::MemoryPool pool(1024 * 1024 * 100);  // 100MB

auto ptr = pool.allocate<float>(M * N);
pool.deallocate(ptr);
```

## 常见问题

### Q: 如何选择 GEMM 版本？

| 场景 | 推荐版本 |
|------|----------|
| 学习/调试 | NAIVE |
| 通用场景 | TILED |
| 大矩阵 | TENSOR_CORE |
| 内存受限 | DOUBLE_BUFFER |

### Q: 编译时找不到头文件？

确保 include 路径正确：

```cmake
target_include_directories(my_app PRIVATE
    ${TENSORCRAFT_ROOT}/include
)
```

### Q: 运行时 CUDA 错误？

1. 检查 GPU 架构是否匹配
2. 使用 `TC_CUDA_CHECK` 定位错误
3. 确保内存已正确分配

## References
