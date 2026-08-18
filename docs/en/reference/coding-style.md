---
outline: [2, 3]
---

# Coding Style

Code style and technical standards for CUDA Kernel Academy.

## C++/CUDA Standards

| Module | Minimum | Recommended |
|--------|---------|-------------|
| 01-sgemm-tutorial | C++17 | C++17 |
| 02-tensorcraft-core | C++17 | C++20 |
| 03-hpc-advanced | C++20 | C++20 |
| 04-inference-engine | C++17 | C++17 |

## Naming Conventions

### Files

```text
cuda_check.hpp    # C++ header (.hpp)
gemm.cuh          # CUDA header (.cuh)
inference_engine.cpp  # C++ source (.cpp)
tiled_gemm.cu     # CUDA source (.cu)
test_gemm.cpp     # Test file (test_ prefix)
```

### Code

```cpp
namespace tensorcraft { }      // snake_case
class InferenceEngine { };       // PascalCase
void launch_gemm(...);          // snake_case
int block_size;                  // snake_case
constexpr int kBlockSize = 32; // kPascalCase or UPPER_SNAKE_CASE
#define TC_CUDA_CHECK(call)    // UPPER_SNAKE_CASE
class MyClass {
    int value_;                 // trailing underscore
};
```

## CUDA Error Handling

```cpp
#include "cuda_foundations/cuda_foundations.hpp"

CUDA_CHECK(cudaMalloc(&ptr, size));
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

## CMake Standards

```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
```

## Pre-Commit Checklist

- [ ] Compiles without warnings
- [ ] Follows naming conventions
- [ ] Uses RAII for resources
- [ ] Adds CUDA error checks
- [ ] Runs `pre-commit run --all-files`
- [ ] Adds unit tests

## References
