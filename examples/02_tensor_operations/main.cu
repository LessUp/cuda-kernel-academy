/**
 * @file main.cu
 * @brief Tensor 操作示例 | Tensor Operations Example
 *
 * 本示例演示基本的 Tensor 操作：
 * 1. Tensor 类的 RAII 内存管理
 * 2. 元素级操作 (加法、乘法、ReLU)
 * 3. Host/Device 数据传输
 *
 * This example demonstrates basic Tensor operations:
 * 1. RAII memory management with Tensor class
 * 2. Element-wise operations (add, multiply, ReLU)
 * 3. Host/Device data transfer
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

// ============================================================================
// CUDA 错误检查宏 | CUDA Error Checking Macro
// ============================================================================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ============================================================================
// 简单的 Tensor 类 | Simple Tensor Class
// ============================================================================
/**
 * @brief 简单的 Tensor 类，演示 RAII 内存管理
 *
 * 这是一个教学用的简化实现，展示了：
 * - 自动内存分配和释放
 * - Host/Device 数据同步
 * - 移动语义
 *
 * This is a simplified implementation for teaching purposes, demonstrating:
 * - Automatic memory allocation and deallocation
 * - Host/Device data synchronization
 * - Move semantics
 */
class Tensor {
public:
    // 构造函数 | Constructor
    Tensor(size_t size) : size_(size), h_data_(nullptr), d_data_(nullptr) {
        // 分配主机内存 | Allocate host memory
        h_data_ = new float[size];

        // 分配设备内存 | Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_data_, size * sizeof(float)));

        printf("Tensor created: size=%zu, host=%p, device=%p\n",
               size, (void*)h_data_, (void*)d_data_);
    }

    // 析构函数 | Destructor
    ~Tensor() {
        if (h_data_) {
            delete[] h_data_;
            h_data_ = nullptr;
        }
        if (d_data_) {
            cudaFree(d_data_);
            d_data_ = nullptr;
        }
        printf("Tensor destroyed: size=%zu\n", size_);
    }

    // 禁用拷贝 | Disable copy
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // 移动构造 | Move constructor
    Tensor(Tensor&& other) noexcept
        : size_(other.size_), h_data_(other.h_data_), d_data_(other.d_data_) {
        other.size_ = 0;
        other.h_data_ = nullptr;
        other.d_data_ = nullptr;
    }

    // 移动赋值 | Move assignment
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            // 释放当前资源 | Release current resources
            delete[] h_data_;
            cudaFree(d_data_);

            // 转移所有权 | Transfer ownership
            size_ = other.size_;
            h_data_ = other.h_data_;
            d_data_ = other.d_data_;

            other.size_ = 0;
            other.h_data_ = nullptr;
            other.d_data_ = nullptr;
        }
        return *this;
    }

    // 访问器 | Accessors
    size_t size() const { return size_; }
    float* host_data() { return h_data_; }
    const float* host_data() const { return h_data_; }
    float* device_data() { return d_data_; }
    const float* device_data() const { return d_data_; }

    // Host -> Device 传输
    void to_device() {
        CUDA_CHECK(cudaMemcpy(d_data_, h_data_, size_ * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Device -> Host 传输
    void to_host() {
        CUDA_CHECK(cudaMemcpy(h_data_, d_data_, size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    // 用常数填充 | Fill with constant
    void fill(float value) {
        for (size_t i = 0; i < size_; ++i) {
            h_data_[i] = value;
        }
    }

    // 用随机数填充 | Fill with random values
    void fill_random(float min = 0.0f, float max = 1.0f) {
        // Fixed-seed generator so runs are reproducible; rand() without seeding
        // gives identical (and unseeded) results every run.
        static std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(min, max);
        for (size_t i = 0; i < size_; ++i) {
            h_data_[i] = dist(gen);
        }
    }

    // 打印前 n 个元素 | Print first n elements
    void print(size_t n = 10) const {
        printf("Tensor[%zu]: ", size_);
        for (size_t i = 0; i < std::min(n, size_); ++i) {
            printf("%.4f ", h_data_[i]);
        }
        if (size_ > n) printf("...");
        printf("\n");
    }

private:
    size_t size_;
    float* h_data_;
    float* d_data_;
};

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief 向量加法 kernel | Vector addition kernel
 * C = A + B
 */
__global__ void add_kernel(const float* A, const float* B, float* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * @brief 向量乘法 kernel | Vector multiplication kernel
 * C = A * B (element-wise)
 */
__global__ void mul_kernel(const float* A, const float* B, float* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}

/**
 * @brief 标量乘法 kernel | Scalar multiplication kernel
 * B = A * scalar
 */
__global__ void scale_kernel(const float* A, float scalar, float* B, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        B[idx] = A[idx] * scalar;
    }
}

/**
 * @brief ReLU 激活函数 kernel | ReLU activation kernel
 * B = max(0, A)
 */
__global__ void relu_kernel(const float* A, float* B, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        B[idx] = fmaxf(0.0f, A[idx]);
    }
}

/**
 * @brief Sigmoid 激活函数 kernel | Sigmoid activation kernel
 * B = 1 / (1 + exp(-A))
 */
__global__ void sigmoid_kernel(const float* A, float* B, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        B[idx] = 1.0f / (1.0f + expf(-A[idx]));
    }
}

// ============================================================================
// 高级操作封装 | High-level Operation Wrappers
// ============================================================================

void tensor_add(const Tensor& A, const Tensor& B, Tensor& C) {
    size_t n = A.size();
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    add_kernel<<<grid_size, block_size>>>(
        A.device_data(), B.device_data(), C.device_data(), n);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_mul(const Tensor& A, const Tensor& B, Tensor& C) {
    size_t n = A.size();
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    mul_kernel<<<grid_size, block_size>>>(
        A.device_data(), B.device_data(), C.device_data(), n);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_scale(const Tensor& A, float scalar, Tensor& B) {
    size_t n = A.size();
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    scale_kernel<<<grid_size, block_size>>>(
        A.device_data(), scalar, B.device_data(), n);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_relu(const Tensor& A, Tensor& B) {
    size_t n = A.size();
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    relu_kernel<<<grid_size, block_size>>>(
        A.device_data(), B.device_data(), n);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_sigmoid(const Tensor& A, Tensor& B) {
    size_t n = A.size();
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    sigmoid_kernel<<<grid_size, block_size>>>(
        A.device_data(), B.device_data(), n);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 主函数 | Main Function
// ============================================================================
int main() {
    printf("=== CUDA Kernel Academy: Tensor Operations Example ===\n\n");

    // 设置随机种子 | Set random seed
    srand(42);

    const size_t N = 1000000;  // 1M 元素
    printf("Tensor size: %zu elements (%.2f MB)\n\n",
           N, N * sizeof(float) / (1024.0f * 1024.0f));

    // ========================================================================
    // 1. 创建 Tensor | Create Tensors
    // ========================================================================
    printf("--- Creating Tensors ---\n");
    Tensor A(N);
    Tensor B(N);
    Tensor C(N);

    // 初始化数据 | Initialize data
    A.fill_random(-1.0f, 1.0f);
    B.fill_random(0.0f, 2.0f);

    printf("\nInitial values:\n");
    printf("A: "); A.print(5);
    printf("B: "); B.print(5);

    // 传输到 GPU | Transfer to GPU
    A.to_device();
    B.to_device();

    // ========================================================================
    // 2. 向量加法 | Vector Addition
    // ========================================================================
    printf("\n--- Vector Addition: C = A + B ---\n");
    tensor_add(A, B, C);
    CUDA_CHECK(cudaDeviceSynchronize());
    C.to_host();
    printf("C: "); C.print(5);

    // 验证 | Verify
    bool correct = true;
    for (size_t i = 0; i < 100; ++i) {
        float expected = A.host_data()[i] + B.host_data()[i];
        if (std::fabs(C.host_data()[i] - expected) > 1e-5f) {
            correct = false;
            break;
        }
    }
    printf("Verification: %s\n", correct ? "PASSED ✓" : "FAILED ✗");

    // ========================================================================
    // 3. 向量乘法 | Vector Multiplication
    // ========================================================================
    printf("\n--- Element-wise Multiplication: C = A * B ---\n");
    tensor_mul(A, B, C);
    CUDA_CHECK(cudaDeviceSynchronize());
    C.to_host();
    printf("C: "); C.print(5);

    // ========================================================================
    // 4. 标量乘法 | Scalar Multiplication
    // ========================================================================
    printf("\n--- Scalar Multiplication: C = A * 2.5 ---\n");
    tensor_scale(A, 2.5f, C);
    CUDA_CHECK(cudaDeviceSynchronize());
    C.to_host();
    printf("C: "); C.print(5);

    // ========================================================================
    // 5. ReLU 激活 | ReLU Activation
    // ========================================================================
    printf("\n--- ReLU Activation: C = max(0, A) ---\n");
    tensor_relu(A, C);
    CUDA_CHECK(cudaDeviceSynchronize());
    C.to_host();
    printf("A: "); A.print(5);
    printf("C: "); C.print(5);

    // 验证 ReLU | Verify ReLU
    correct = true;
    for (size_t i = 0; i < 100; ++i) {
        float expected = std::fmax(0.0f, A.host_data()[i]);
        if (std::fabs(C.host_data()[i] - expected) > 1e-5f) {
            correct = false;
            break;
        }
    }
    printf("Verification: %s\n", correct ? "PASSED ✓" : "FAILED ✗");

    // ========================================================================
    // 6. Sigmoid 激活 | Sigmoid Activation
    // ========================================================================
    printf("\n--- Sigmoid Activation: C = 1/(1+exp(-A)) ---\n");
    tensor_sigmoid(A, C);
    CUDA_CHECK(cudaDeviceSynchronize());
    C.to_host();
    printf("A: "); A.print(5);
    printf("C: "); C.print(5);

    // ========================================================================
    // 7. 演示移动语义 | Demonstrate Move Semantics
    // ========================================================================
    printf("\n--- Move Semantics Demo ---\n");
    {
        Tensor temp(100);
        temp.fill(3.14f);
        printf("Before move - temp: "); temp.print(5);

        Tensor moved = std::move(temp);
        printf("After move - moved: "); moved.print(5);
        // temp 现在是空的 | temp is now empty
    }
    printf("Tensors destroyed when leaving scope\n");

    // ========================================================================
    // 8. 性能测试 | Performance Test
    // ========================================================================
    printf("\n--- Performance Test ---\n");

    // Warmup
    for (int i = 0; i < 10; ++i) {
        tensor_add(A, B, C);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        tensor_add(A, B, C);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;

    // 计算带宽 | Calculate bandwidth
    // 读取 2 个数组，写入 1 个数组
    float bytes = 3.0f * N * sizeof(float);
    float bandwidth = bytes / (avg_ms * 1e6f);  // GB/s

    printf("Vector Add (%zu elements):\n", N);
    printf("  Average time: %.3f ms\n", avg_ms);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\n=== Example Complete ===\n");
    return 0;
}
