# 故障排除指南

本文档收集了 CUDA Kernel Academy 项目的常见问题及解决方案。

## 目录

- [编译错误](#编译错误)
- [运行时错误](#运行时错误)
- [性能问题](#性能问题)
- [CUDA 版本兼容性](#cuda-版本兼容性)
- [CMake 配置问题](#cmake-配置问题)
- [测试失败](#测试失败)

---

## 编译错误

### 1. nvcc: command not found

**症状**：

```
nvcc: command not found
```

**原因**：CUDA Toolkit 未安装或未添加到 PATH。

**解决方案**：

```bash
# 检查 CUDA 安装
ls /usr/local/cuda

# 添加到 PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 永久生效
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 2. unsupported GNU version

**症状**：

```
#error -- unsupported GNU version! gcc versions later than X are not supported!
```

**原因**：GCC 版本与 CUDA 版本不兼容。

**解决方案**：

| CUDA 版本 | 支持的 GCC 版本 |
|-----------|-----------------|
| CUDA 11.x | GCC ≤ 10 |
| CUDA 12.0-12.2 | GCC ≤ 12 |
| CUDA 12.3+ | GCC ≤ 13 |

```bash
# 安装兼容版本的 GCC
sudo apt-get install gcc-11 g++-11

# 使用特定版本编译
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
         -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11
```

### 3. no kernel image is available for execution

**症状**：

```
CUDA error: no kernel image is available for execution on the device
```

**原因**：编译的 GPU 架构与运行的 GPU 不匹配。

**解决方案**：

```bash
# 查询 GPU 计算能力
nvidia-smi --query-gpu=compute_cap --format=csv

# 重新编译，指定正确的架构
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86  # RTX 3090
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80  # A100
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90  # H100
```

### 4. undefined reference to `cudaXXX`

**症状**：

```
undefined reference to `cudaMalloc'
undefined reference to `cudaFree'
```

**原因**：未链接 CUDA 运行时库。

**解决方案**：

在 CMakeLists.txt 中添加：

```cmake
find_package(CUDAToolkit REQUIRED)
target_link_libraries(your_target PRIVATE CUDA::cudart)
```

### 5. error: identifier "half" is undefined

**症状**：

```
error: identifier "half" is undefined
error: identifier "__half" is undefined
```

**原因**：未包含 CUDA FP16 头文件。

**解决方案**：

```cpp
#include <cuda_fp16.h>

// 或使用完整路径
#include <cuda/std/cstdint>
```

### 6. ptxas error: Entry function uses too much shared memory

**症状**：

```
ptxas error: Entry function 'kernel_name' uses too much shared memory
```

**原因**：共享内存使用超过硬件限制。

**解决方案**：

```cpp
// 检查共享内存限制
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Shared memory per block: %zu\n", prop.sharedMemPerBlock);

// 减少共享内存使用或使用动态共享内存
extern __shared__ float shared_mem[];
kernel<<<grid, block, shared_mem_size>>>();
```

---

## 运行时错误

### 1. CUDA error: out of memory

**症状**：

```
CUDA error: out of memory
```

**原因**：GPU 显存不足。

**解决方案**：

```bash
# 检查显存使用
nvidia-smi

# 释放显存
# 1. 关闭其他 GPU 程序
# 2. 减小 batch size 或矩阵大小
# 3. 使用 cudaDeviceReset() 重置设备
```

```cpp
// 在代码中检查显存
size_t free_mem, total_mem;
cudaMemGetInfo(&free_mem, &total_mem);
printf("Free: %zu MB, Total: %zu MB\n", free_mem/1024/1024, total_mem/1024/1024);
```

### 2. CUDA error: illegal memory access

**症状**：

```
CUDA error: an illegal memory access was encountered
```

**原因**：访问了无效的 GPU 内存地址。

**解决方案**：

```bash
# 使用 compute-sanitizer 调试
compute-sanitizer --tool memcheck ./your_program
```

常见原因：

- 数组越界访问
- 使用已释放的内存
- 主机指针传给 GPU kernel
- 线程索引计算错误

### 3. CUDA error: misaligned address

**症状**：

```
CUDA error: misaligned address
```

**原因**：内存访问未对齐。

**解决方案**：

```cpp
// 确保内存对齐
float* d_ptr;
cudaMalloc(&d_ptr, size);  // cudaMalloc 返回 256 字节对齐的地址

// 使用对齐的数据类型
struct __align__(16) AlignedStruct {
    float4 data;
};
```

### 4. CUDA error: device-side assert triggered

**症状**：

```
CUDA error: device-side assert triggered
```

**原因**：GPU 代码中的 assert 失败。

**解决方案**：

```bash
# 启用设备端调试
export CUDA_LAUNCH_BLOCKING=1

# 重新运行程序查看详细错误位置
./your_program
```

### 5. Segmentation fault (core dumped)

**症状**：程序崩溃，无 CUDA 错误信息。

**原因**：主机端代码错误或 CUDA 错误未检查。

**解决方案**：

```cpp
// 始终检查 CUDA 错误
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 同步并检查 kernel 错误
kernel<<<grid, block>>>();
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

---

## 性能问题

### 1. Kernel 运行很慢

**诊断**：

```bash
# 使用 nsys 分析
nsys profile --stats=true ./your_program

# 使用 ncu 分析 kernel
ncu --set full ./your_program
```

**常见原因和解决方案**：

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| 内存访问不合并 | 低内存带宽利用率 | 确保连续线程访问连续内存 |
| Bank Conflict | 共享内存带宽低 | 使用 padding 或重新组织数据 |
| Warp Divergence | 低 SM 利用率 | 减少分支，使用 warp-level 原语 |
| 寄存器溢出 | 高 local memory 使用 | 减少局部变量，使用 `__launch_bounds__` |
| Occupancy 低 | SM 利用率低 | 调整 block size，减少资源使用 |

### 2. 内存传输瓶颈

**诊断**：

```bash
# 检查 PCIe 带宽
nvidia-smi topo -m
```

**解决方案**：

```cpp
// 使用 pinned memory
float* h_data;
cudaMallocHost(&h_data, size);  // 比 malloc 快 2-3x

// 使用异步传输
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);

// 重叠计算和传输
cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1);
kernel<<<grid, block, 0, stream2>>>(d_b);
```

### 3. GPU 利用率低

**诊断**：

```bash
# 监控 GPU 利用率
nvidia-smi dmon -s u
```

**解决方案**：

- 增加并行度（更大的 grid/block）
- 使用多 stream 并发执行
- 减少 CPU-GPU 同步点
- 使用 CUDA Graphs 减少 launch overhead

---

## CUDA 版本兼容性

### CUDA 11.x vs 12.x 差异

| 特性 | CUDA 11.x | CUDA 12.x |
|------|-----------|-----------|
| C++ 标准 | C++14/17 | C++17/20 |
| PTX 版本 | 7.x | 8.x |
| 新架构支持 | sm_80, sm_86 | sm_89, sm_90 |
| TMA | ❌ | ✅ (sm_90) |
| Thread Block Cluster | ❌ | ✅ (sm_90) |

### 向后兼容性

```cmake
# 支持多 CUDA 版本
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
    target_compile_definitions(your_target PRIVATE CUDA_12_OR_LATER)
endif()
```

```cpp
#if __CUDA_ARCH__ >= 900
    // Hopper 特定代码
#elif __CUDA_ARCH__ >= 800
    // Ampere 特定代码
#else
    // 通用代码
#endif
```

---

## CMake 配置问题

### 1. Could not find CUDA

**症状**：

```
CMake Error: Could not find CUDA
```

**解决方案**：

```bash
# 设置 CUDA 路径
export CUDA_HOME=/usr/local/cuda
cmake .. -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
```

### 2. CMake 版本过低

**症状**：

```
CMake 3.20 or higher is required
```

**解决方案**：

```bash
# Ubuntu: 使用 snap 安装最新版
sudo snap install cmake --classic

# 或从源码安装
wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.sh
chmod +x cmake-3.28.0-linux-x86_64.sh
sudo ./cmake-3.28.0-linux-x86_64.sh --prefix=/usr/local --skip-license
```

### 3. Ninja not found

**症状**：

```
CMake Error: CMake was unable to find a build program corresponding to "Ninja"
```

**解决方案**：

```bash
# 安装 Ninja
sudo apt-get install ninja-build  # Ubuntu
sudo dnf install ninja-build      # Fedora

# 或使用 Make
cmake .. -G "Unix Makefiles"
```

---

## 测试失败

### 1. 数值精度问题

**症状**：

```
Expected: 1.0
Actual: 0.99999994
```

**解决方案**：

```cpp
// 使用适当的容差
EXPECT_NEAR(expected, actual, 1e-5);

// 或使用相对误差
float rel_error = std::abs(expected - actual) / std::abs(expected);
EXPECT_LT(rel_error, 1e-4);
```

### 2. 随机测试失败

**症状**：测试有时通过，有时失败。

**原因**：

- 未初始化的内存
- 竞态条件
- 随机数种子问题

**解决方案**：

```cpp
// 初始化所有内存
cudaMemset(d_ptr, 0, size);

// 使用固定的随机种子
srand(42);
curandSetPseudoRandomGeneratorSeed(gen, 42);

// 添加同步点
cudaDeviceSynchronize();
```

### 3. 测试超时

**症状**：

```
Test timeout after 300 seconds
```

**解决方案**：

```cmake
# 增加测试超时时间
set_tests_properties(your_test PROPERTIES TIMEOUT 600)
```

```bash
# 或在命令行指定
ctest --timeout 600
```

---

## 获取更多帮助

如果以上方案无法解决问题：

1. 搜索 [GitHub Issues](https://github.com/LessUp/cuda-kernel-academy/issues)
2. 查看 [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
3. 创建新 Issue，提供：
   - 完整的错误信息
   - 操作系统和 CUDA 版本
   - GPU 型号
   - 最小可复现代码

---

## 调试工具推荐

| 工具 | 用途 | 命令 |
|------|------|------|
| compute-sanitizer | 内存错误检测 | `compute-sanitizer ./program` |
| cuda-gdb | GPU 调试器 | `cuda-gdb ./program` |
| nsys | 系统级性能分析 | `nsys profile ./program` |
| ncu | Kernel 级性能分析 | `ncu ./program` |
| nvprof | 旧版性能分析 (CUDA 11-) | `nvprof ./program` |
