# 快速入门指南

本指南帮助你快速上手 Mini-Inference Engine。

## 环境要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+)
- **CUDA**: 11.0+ (推荐 12.0+)
- **GPU**: NVIDIA GPU，Compute Capability 7.5+
- **编译器**: GCC 9+ 或 Clang 10+
- **CMake**: 3.18+

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/mini-inference-engine.git
cd mini-inference-engine
```

### 2. 构建项目

```bash
# 创建构建目录
mkdir build && cd build

# 配置 (不含测试，更快)
cmake -DBUILD_TESTS=OFF ..

# 或者包含测试
cmake ..

# 编译
make -j$(nproc)
```

### 3. 验证安装

```bash
# 运行 benchmark
./benchmark

# 如果看到性能数据输出，说明安装成功
```

## 第一个程序

### 使用 GEMM Kernel

```cpp
#include "common.h"
#include "kernels.cuh"

int main() {
    // 初始化 CUDA
    CUDA_CHECK(cudaSetDevice(0));
    
    // 矩阵维度
    int M = 1024, N = 1024, K = 1024;
    
    // 分配内存
    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));
    
    // 初始化数据
    std::vector<float> h_A(M * K), h_B(K * N);
    random_init(h_A.data(), h_A.size());
    random_init(h_B.data(), h_B.size());
    
    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));
    
    // 执行 GEMM
    launch_optimized_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    
    // 同步
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 获取结果
    std::vector<float> h_C(M * N);
    d_C.copy_to_host(h_C.data(), M * N * sizeof(float));
    
    printf("GEMM completed! C[0] = %f\n", h_C[0]);
    
    return 0;
}
```

### 编译运行

```bash
# 编译
nvcc -o my_gemm my_gemm.cu -I../include -L../build -lmini_inference -lcudart -lcublas

# 运行
./my_gemm
```

## 使用推理引擎

### 基本用法

```cpp
#include "inference_engine.h"

int main() {
    // 创建引擎
    InferenceEngine engine;
    engine.init(0);  // GPU 0
    
    // 加载权重
    if (!engine.load_weights("model.bin")) {
        printf("Failed to load weights!\n");
        return 1;
    }
    
    // 准备输入
    int batch_size = 32;
    int input_dim = engine.input_dim();
    int output_dim = engine.output_dim();
    
    DeviceMemory d_input(batch_size * input_dim * sizeof(float));
    DeviceMemory d_output(batch_size * output_dim * sizeof(float));
    
    // 初始化输入数据
    std::vector<float> input(batch_size * input_dim);
    // ... 填充输入数据 ...
    d_input.copy_from_host(input.data(), input.size() * sizeof(float));
    
    // 推理
    engine.forward(d_input.get(), d_output.get(), batch_size);
    
    // 获取输出
    std::vector<float> output(batch_size * output_dim);
    d_output.copy_to_host(output.data(), output.size() * sizeof(float));
    
    // 清理
    engine.cleanup();
    
    return 0;
}
```

### 创建权重文件

```cpp
#include "inference_engine.h"

// 创建随机权重用于测试
std::vector<std::pair<int, int>> layer_dims = {
    {784, 256},   // Layer 0: 784 -> 256
    {256, 128},   // Layer 1: 256 -> 128
    {128, 10}     // Layer 2: 128 -> 10
};

create_random_weights("model.bin", layer_dims, true);  // true = 包含 bias
```

## 使用 Tensor 类

```cpp
#include "tensor.h"

int main() {
    CUDA_CHECK(cudaSetDevice(0));
    
    // 创建 tensor
    Tensor a({32, 784});  // batch_size=32, features=784
    Tensor w({784, 256}); // weight matrix
    Tensor b({256});      // bias
    
    // 初始化
    std::vector<float> a_data(32 * 784);
    random_init(a_data.data(), a_data.size());
    a.copy_from_host(a_data.data());
    
    // 操作
    Tensor c = matmul(a, w);           // 矩阵乘法
    Tensor d = add_bias(c, b);         // 加 bias
    Tensor e = relu(d);                // ReLU 激活
    
    // 或者使用融合操作
    Tensor f = fused_linear_relu(a, w, b);
    
    // 获取结果
    auto result = f.to_host();
    
    return 0;
}
```

## 性能测量

```cpp
#include "kernels.cuh"

int main() {
    // ... 初始化 ...
    
    // 使用 GpuTimer
    GpuTimer timer;
    
    timer.start();
    launch_optimized_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    timer.stop();
    
    float ms = timer.elapsed_ms();
    float gflops = 2.0f * M * N * K / (ms * 1e6);
    
    printf("Time: %.3f ms, GFLOPS: %.2f\n", ms, gflops);
    
    // 使用 benchmark_kernel
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    PerfStats stats = benchmark_kernel(
        GemmKernelType::REGISTER_BLOCKED,
        d_A.get(), d_B.get(), d_C.get(),
        M, N, K,
        5,    // warmup iterations
        20,   // benchmark iterations
        handle
    );
    
    printf("Avg Time: %.3f ms, GFLOPS: %.2f, vs cuBLAS: %.1f%%\n",
           stats.kernel_time_ms, stats.gflops, stats.cublas_ratio * 100);
    
    cublasDestroy(handle);
    
    return 0;
}
```

## 配置系统

### 使用配置文件

```cpp
#include "config.h"

int main() {
    // 加载配置
    Config::instance().load_from_file("config/default.ini");
    
    // 读取配置
    int device = Config::instance().get_int("CUDA_DEVICE", 0);
    bool use_tensor_cores = Config::instance().get_bool("ENABLE_TENSOR_CORES", true);
    std::string preset = Config::instance().get("GEMM_PRESET", "medium");
    
    // 使用 GEMM 预设
    GemmPreset gemm_config = get_gemm_preset(preset);
    printf("Using preset: %s (BM=%d, BN=%d)\n", 
           gemm_config.name.c_str(), gemm_config.block_m, gemm_config.block_n);
    
    return 0;
}
```

### 配置文件示例

```ini
# config/my_config.ini
CUDA_DEVICE=0
GEMM_PRESET=large
ENABLE_TENSOR_CORES=true
LOG_LEVEL=INFO
```

## 日志系统

```cpp
#include "logger.h"

int main() {
    // 设置日志级别
    Logger::instance().set_level(LogLevel::DEBUG);
    
    // 输出到文件
    Logger::instance().set_file("app.log");
    
    // 使用日志宏
    LOG_INFO("Starting application");
    LOG_DEBUG("Matrix size: %d x %d", M, N);
    LOG_WARN("Performance below target");
    LOG_ERROR("CUDA error occurred");
    
    return 0;
}
```

## 常见问题

### Q: 编译错误 "Unsupported gpu architecture"

A: 修改 CMakeLists.txt 中的 `CMAKE_CUDA_ARCHITECTURES`:
```cmake
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
```

### Q: 运行时 "CUDA out of memory"

A: 减小矩阵大小或 batch size，或使用内存池:
```cpp
MemoryPool::instance().clear_cache();
```

### Q: 性能低于预期

A: 检查:
1. GPU 是否在 P0 状态: `nvidia-smi -q -d PERFORMANCE`
2. 是否有其他程序占用 GPU
3. 矩阵大小是否太小

### Q: 数值精度问题

A: 使用 CPU 参考实现验证:
```cpp
std::vector<float> c_ref(M * N);
cpu_matmul(h_A.data(), h_B.data(), c_ref.data(), M, N, K);
float error = compare_matrices(h_C.data(), c_ref.data(), M * N);
printf("Max error: %e\n", error);
```

## 下一步

- 阅读 [架构设计文档](ARCHITECTURE.md)
- 学习 [GEMM 优化详解](GEMM_OPTIMIZATION.md)
- 参考 [性能调优指南](PERFORMANCE_TUNING.md)
- 查看 [API 参考](API_REFERENCE.md)
