# Modern C++ for CUDA Development

This guide demonstrates how to leverage modern C++ features (C++17/20/23) in CUDA kernel development.

## C++ Version Detection

TensorCraft-HPC automatically detects and uses the highest available C++ standard:

```cpp
// include/tensorcraft/core/features.hpp
#if __cplusplus >= 202302L
    #define TC_CPP23 1
#endif
#if __cplusplus >= 202002L
    #define TC_CPP20 1
#endif
#if __cplusplus >= 201703L
    #define TC_CPP17 1
#endif
```

## C++20 Concepts for Type Constraints

Instead of complex SFINAE, use Concepts to constrain template parameters:

```cpp
// C++20 Concepts
template<typename T>
concept Numeric = std::is_arithmetic_v<T> || is_half_v<T>;

template<typename T>
concept FloatingPoint = std::is_floating_point_v<T> || is_half_v<T>;

// Usage in kernel
template<Numeric T>
__global__ void my_kernel(T* data, size_t n) {
    // T is guaranteed to be a numeric type
}
```

With C++17 fallback using SFINAE:

```cpp
// C++17 SFINAE version
template<typename T, typename = void>
struct is_numeric : std::false_type {};

template<typename T>
struct is_numeric<T, std::enable_if_t<
    std::is_arithmetic_v<T> || is_half_v<T>>> : std::true_type {};

template<typename T>
inline constexpr bool is_numeric_v = is_numeric<T>::value;
```

## Constexpr for Compile-Time Computation

Use `constexpr` to compute kernel launch parameters at compile time:

```cpp
// Compile-time block size selection
template<typename T>
constexpr int optimal_block_size() {
    if constexpr (sizeof(T) <= 2) {
        return 512;  // More threads for smaller types
    } else {
        return 256;
    }
}

// Compile-time vector size selection
template<typename T>
constexpr int optimal_vec_size() {
    if constexpr (sizeof(T) == 1) return 16;
    else if constexpr (sizeof(T) == 2) return 8;
    else if constexpr (sizeof(T) == 4) return 4;
    else return 2;
}
```

## Generic Elementwise Kernels with Functors

Use functors and lambdas for flexible kernel composition:

```cpp
// Functor pattern
struct ReLU {
    template<typename T>
    __device__ __forceinline__ T operator()(T x) const {
        return x > T(0) ? x : T(0);
    }
};

struct GeLU {
    template<typename T>
    __device__ __forceinline__ T operator()(T x) const {
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;
        constexpr float coeff = 0.044715f;
        float xf = static_cast<float>(x);
        float inner = sqrt_2_over_pi * (xf + coeff * xf * xf * xf);
        return static_cast<T>(0.5f * xf * (1.0f + tanhf(inner)));
    }
};

// Generic kernel
template<typename T, typename Func, int VecSize = 4>
__global__ void elementwise_kernel(const T* input, T* output, size_t n, Func func) {
    using VecT = AlignedVector<T, VecSize>;
    
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
    
    if (idx + VecSize <= n) {
        VecT in_vec = *reinterpret_cast<const VecT*>(&input[idx]);
        VecT out_vec;
        
        #pragma unroll
        for (int k = 0; k < VecSize; ++k) {
            out_vec[k] = func(in_vec[k]);
        }
        
        *reinterpret_cast<VecT*>(&output[idx]) = out_vec;
    }
}

// Usage
launch_elementwise(input, output, n, ReLU{});
launch_elementwise(input, output, n, GeLU{});
```

## Structured Bindings (C++17)

Use structured bindings for cleaner code:

```cpp
// Return multiple values
struct KernelConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem;
};

KernelConfig compute_config(size_t n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    return {dim3(grid_size), dim3(block_size), 0};
}

// Usage with structured bindings
auto [grid, block, smem] = compute_config(n);
my_kernel<<<grid, block, smem>>>(data, n);
```

## std::optional for Optional Parameters (C++17)

```cpp
template<typename T>
void launch_layernorm(
    const T* input,
    const T* gamma,
    const T* beta,  // Can be nullptr
    T* output,
    int batch_size,
    int hidden_size,
    std::optional<float> eps = std::nullopt) {
    
    float epsilon = eps.value_or(1e-5f);
    // ...
}
```

## if constexpr for Compile-Time Branching (C++17)

```cpp
template<typename T>
__device__ __forceinline__ float to_float(T val) {
    if constexpr (std::is_same_v<T, float>) {
        return val;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __half2float(val);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(val);
    } else {
        return static_cast<float>(val);
    }
}
```

## Inline Variables (C++17)

```cpp
// Header-only constants
template<typename T>
inline constexpr T pi = T(3.14159265358979323846);

template<typename T>
inline constexpr T sqrt_2_over_pi = T(0.7978845608028654);
```

## RAII for CUDA Resources

```cpp
template<typename T>
class Tensor {
public:
    explicit Tensor(const std::vector<size_t>& shape) 
        : shape_(shape), size_(compute_size(shape)) {
        TC_CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(T)));
    }
    
    ~Tensor() {
        if (data_) cudaFree(data_);
    }
    
    // Move semantics
    Tensor(Tensor&& other) noexcept 
        : data_(other.data_), shape_(std::move(other.shape_)), size_(other.size_) {
        other.data_ = nullptr;
    }
    
    // Disable copy
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
private:
    T* data_ = nullptr;
    std::vector<size_t> shape_;
    size_t size_ = 0;
};
```

## Best Practices

1. **Use `constexpr` liberally** - Move computations to compile time
2. **Prefer Concepts over SFINAE** - Clearer error messages, easier to read
3. **Use `if constexpr`** - Eliminate runtime branches for type-dependent code
4. **Leverage RAII** - Automatic resource management prevents leaks
5. **Use structured bindings** - Cleaner code when returning multiple values
6. **Prefer `std::optional`** - Explicit handling of optional values
