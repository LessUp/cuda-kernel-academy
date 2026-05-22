---
outline: [2, 3]
---

# Integration Examples

How to integrate CUDA Kernel Academy modules into your project.

## Using TensorCraft Core

### As Subdirectory

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_cuda_project LANGUAGES CXX CUDA)

add_subdirectory(path/to/02-tensorcraft-core tensorcraft)
add_executable(my_app main.cu)
target_link_libraries(my_app PRIVATE tensorcraft)
```

### Header-Only

```cmake
target_include_directories(my_app PRIVATE
    path/to/02-tensorcraft-core/include
)
```

## GEMM Example

```cpp
#include <tensorcraft/kernels/gemm.hpp>

int main() {
    tensorcraft::kernels::launch_gemm(
        d_A, d_B, d_C, M, N, K,
        1.0f, 0.0f,
        tensorcraft::kernels::GemmVersion::TILED
    );
    return 0;
}
```

## References
