---
outline: [2, 3]
---

# TensorCraft Core API Reference

Complete API reference for the TensorCraft Core library.

## Core Module

### cuda_check.hpp

```cpp
TC_CUDA_CHECK(err)      // Check CUDA error, throw on failure
TC_CUDA_CHECK_LAST()    // Check last CUDA error
```

### features.hpp

```cpp
TC_CPP17 / TC_CPP20 / TC_CPP23
TC_CUDA_VERSION
TC_HAS_WMMA / TC_HAS_FP16 / TC_HAS_BF16 / TC_HAS_FP8
```

## Memory Module

### AlignedVector

```cpp
template<typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector;
```

### Tensor

```cpp
template<typename T>
class Tensor { ... };
```

### MemoryPool

```cpp
class MemoryPool {
    static MemoryPool& instance();
    void* allocate(size_t bytes);
    void deallocate(void* ptr);
};
```

## Kernels Module

See the Chinese version for full API details: [API Reference](../../zh/modules/02/api-reference.md)

## References

[^1]: NVIDIA. "CUDA C++ Programming Guide," v12.6. 2024. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
[^2]: pybind11 Documentation. https://pybind11.readthedocs.io/
