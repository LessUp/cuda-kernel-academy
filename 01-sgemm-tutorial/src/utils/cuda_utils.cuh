#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>
#include <random>

// ============================================================================
// Error Checking Macros
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = call;                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                    static_cast<int>(status));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CURAND_CHECK(call)                                                     \
    do {                                                                       \
        curandStatus_t status = call;                                          \
        if (status != CURAND_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuRAND error at %s:%d: %d\n", __FILE__, __LINE__, \
                    static_cast<int>(status));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ============================================================================
// RAII Wrappers for Device Memory
// ============================================================================

template <typename T>
class DeviceMemory {
public:
    DeviceMemory() : ptr_(nullptr), size_(0) {}

    explicit DeviceMemory(size_t count) : size_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }

    ~DeviceMemory() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    // Move semantics
    DeviceMemory(DeviceMemory&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Disable copy
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }

    void copyFromHost(const T* host_ptr, size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr_, host_ptr, count * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    void copyToHost(T* host_ptr, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host_ptr, ptr_, count * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }

    void zero() {
        CUDA_CHECK(cudaMemset(ptr_, 0, size_ * sizeof(T)));
    }

private:
    T* ptr_;
    size_t size_;
};

// ============================================================================
// RAII Wrapper for cuBLAS Handle
// ============================================================================

class CublasHandle {
public:
    CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle_));
    }

    ~CublasHandle() {
        if (handle_) {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() { return handle_; }

private:
    cublasHandle_t handle_;
};

// ============================================================================
// Matrix Initialization Functions
// ============================================================================

// Initialize matrix with random values on host
inline void initRandomMatrix(float* data, int rows, int cols,
                             float min_val = -1.0f, float max_val = 1.0f,
                             unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    for (int i = 0; i < rows * cols; ++i) {
        data[i] = dist(gen);
    }
}

// Initialize matrix with zeros
inline void initZeroMatrix(float* data, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        data[i] = 0.0f;
    }
}

// Initialize identity matrix
inline void initIdentityMatrix(float* data, int n) {
    for (int i = 0; i < n * n; ++i) {
        data[i] = 0.0f;
    }
    for (int i = 0; i < n; ++i) {
        data[i * n + i] = 1.0f;
    }
}

// Initialize matrix with constant value
inline void initConstantMatrix(float* data, int rows, int cols, float val) {
    for (int i = 0; i < rows * cols; ++i) {
        data[i] = val;
    }
}

// ============================================================================
// GPU Random Initialization (using cuRAND)
// ============================================================================

inline void initRandomMatrixGPU(float* d_data, int rows, int cols,
                                float min_val = -1.0f, float max_val = 1.0f,
                                unsigned long long seed = 42) {
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
    
    // Generate uniform [0, 1)
    CURAND_CHECK(curandGenerateUniform(gen, d_data, rows * cols));
    
    // Scale to [min_val, max_val]
    float scale = max_val - min_val;
    // Simple kernel to scale values
    // For simplicity, we'll do this on CPU for now
    
    curandDestroyGenerator(gen);
}

// ============================================================================
// Utility Functions
// ============================================================================

// Print matrix (for debugging)
inline void printMatrix(const float* data, int rows, int cols,
                        const char* name = "Matrix") {
    printf("%s (%d x %d):\n", name, rows, cols);
    for (int i = 0; i < rows && i < 8; ++i) {
        for (int j = 0; j < cols && j < 8; ++j) {
            printf("%8.4f ", data[i * cols + j]);
        }
        if (cols > 8) printf("...");
        printf("\n");
    }
    if (rows > 8) printf("...\n");
    printf("\n");
}

// Get GPU device properties
inline void printGPUInfo() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("GPU Device: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM Count: %d\n", prop.multiProcessorCount);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    // Note: memoryClockRate is deprecated in newer CUDA versions
    // Peak bandwidth calculation uses bus width only as approximation
    printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    printf("\n");
}

// Check if dimensions are valid (multiples of 32)
inline bool isValidDimension(int dim, int alignment = 32) {
    return dim > 0 && (dim % alignment == 0);
}

// Round up to multiple
inline int roundUp(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}
