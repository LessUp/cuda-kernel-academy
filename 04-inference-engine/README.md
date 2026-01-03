# 🧠 04-inference-engine

<p align="center">
  <b>轻量级 CUDA 推理引擎框架</b><br>
  <i>A Lightweight CUDA Inference Engine Framework</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Level-4%20Application-brightgreen?style=flat-square" alt="Level">
  <img src="https://img.shields.io/badge/CUDA-11.0%2B-76B900?style=flat-square&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/C++-17-00599C?style=flat-square&logo=cplusplus" alt="C++">
  <img src="https://img.shields.io/badge/Depends-tensorcraft--core-orange?style=flat-square" alt="Dependency">
</p>

---

## 📍 在学习路径中的位置

```
01-sgemm-tutorial ──▶ 02-tensorcraft-core ──┬──▶ 03-hpc-advanced
                                            │
                                            └──▶ 04-inference-engine ◀── 你在这里
```

**前置要求**：
- ✅ 完成 `01-sgemm-tutorial` 理解 GEMM 优化基础
- ✅ 完成 `02-tensorcraft-core` 理解算子库设计

## 📖 项目简介

**04-inference-engine** 是一个完整的轻量级推理引擎框架，展示如何将底层 CUDA 算子组装成可用的推理系统。本模块依赖 `02-tensorcraft-core` 提供的高性能算子，专注于推理引擎的架构设计。

### 核心特性

- 🔧 **Memory Pool** - 高效的 GPU 内存池管理
- 🌊 **Stream Manager** - 多流并发执行管理
- ⚡ **Auto-tuner** - 自动选择最优 kernel 配置
- 📊 **Profiler** - 性能分析和瓶颈定位
- 🔢 **Quantization** - INT8/FP16 量化支持
- 🧪 **MNIST Demo** - 完整的端到端示例

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                     Inference Engine API                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Tensor    │  │   Config    │  │   Logger    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Memory Pool │  │Stream Mgr   │  │ Auto-tuner  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                    02-tensorcraft-core                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │    GEMM     │  │  Attention  │  │ Elementwise │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 目录结构

```
04-inference-engine/
├── include/                    # 头文件
│   ├── inference_engine.h      # 主引擎接口
│   ├── tensor.h                # Tensor 抽象
│   ├── memory_pool.h           # 内存池
│   ├── stream_manager.h        # 流管理器
│   ├── autotuner.h             # 自动调优
│   ├── profiler.h              # 性能分析
│   ├── quantization.h          # 量化支持
│   ├── config.h                # 配置管理
│   └── logger.h                # 日志系统
├── src/                        # 源文件
│   ├── inference_engine.cpp    # 引擎实现
│   ├── tensor.cu               # Tensor 实现
│   └── ...
├── benchmarks/                 # 性能测试
│   ├── benchmark.cpp           # 通用 benchmark
│   ├── detailed_benchmark.cu   # 详细性能分析
│   └── mnist_demo.cpp          # MNIST 演示
├── tests/                      # 单元测试
├── config/                     # 配置文件
│   ├── default.ini             # 默认配置
│   ├── debug.ini               # 调试配置
│   └── high_performance.ini    # 高性能配置
├── docs/                       # 文档
│   ├── ARCHITECTURE.md         # 架构设计
│   ├── API_REFERENCE.md        # API 参考
│   ├── PERFORMANCE_TUNING.md   # 性能调优
│   └── QUICK_START.md          # 快速开始
├── scripts/                    # 工具脚本
├── CMakeLists.txt              # CMake 配置
├── CHANGELOG.md                # 更新日志
└── README.md
```

## 🔗 依赖关系

本模块依赖 `02-tensorcraft-core` 提供的高性能算子：

```cmake
# CMakeLists.txt 中的依赖配置
add_subdirectory(${CMAKE_SOURCE_DIR}/../02-tensorcraft-core tensorcraft)
target_link_libraries(mini_inference tensorcraft)
```

### 使用的 tensorcraft 组件

| 组件 | 用途 |
|------|------|
| `tensorcraft::kernels::gemm` | 矩阵乘法 |
| `tensorcraft::kernels::elementwise` | 激活函数 |
| `tensorcraft::kernels::softmax` | Softmax |
| `tensorcraft::memory::MemoryPool` | 内存管理参考 |

## 🛠️ 构建

### 前置条件

```bash
# 确保已构建 tensorcraft-core
cd ../02-tensorcraft-core
mkdir build && cd build
cmake .. && make -j$(nproc)
```

### 构建推理引擎

```bash
cd 04-inference-engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### CMake 选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `BUILD_TESTS` | ON | 构建测试 |
| `BUILD_BENCHMARKS` | ON | 构建 benchmark |
| `ENABLE_PROFILING` | OFF | 启用性能分析 |
| `USE_TENSORCRAFT` | ON | 使用 tensorcraft 算子 |

## 🚀 快速开始

### 基本使用

```cpp
#include "inference_engine.h"
#include "tensor.h"

int main() {
    // 创建推理引擎
    InferenceEngine engine;
    engine.loadConfig("config/default.ini");
    
    // 创建输入 Tensor
    Tensor input({1, 784}, DataType::FLOAT32);
    input.loadFromFile("mnist_input.bin");
    
    // 执行推理
    Tensor output = engine.forward(input);
    
    // 获取结果
    int predicted_class = output.argmax();
    std::cout << "Predicted: " << predicted_class << std::endl;
    
    return 0;
}
```

### 使用 Memory Pool

```cpp
#include "memory_pool.h"

MemoryPool pool(1024 * 1024 * 100);  // 100MB

// 分配内存
void* ptr1 = pool.allocate(1024 * 1024);  // 1MB
void* ptr2 = pool.allocate(512 * 1024);   // 512KB

// 释放内存
pool.deallocate(ptr1);
pool.deallocate(ptr2);

// 查看统计
pool.printStats();
```

### 使用 Stream Manager

```cpp
#include "stream_manager.h"

StreamManager streams(4);  // 4 个 CUDA 流

// 在不同流上执行
streams.submit(0, [&]() { kernel1<<<...>>>(...); });
streams.submit(1, [&]() { kernel2<<<...>>>(...); });

// 同步所有流
streams.synchronizeAll();
```

## 📊 性能特性

### Memory Pool 优势

| 场景 | cudaMalloc | Memory Pool | 提升 |
|------|------------|-------------|------|
| 小块频繁分配 | 100μs/次 | 0.1μs/次 | 1000x |
| 大块分配 | 1ms/次 | 10μs/次 | 100x |
| 内存碎片 | 高 | 低 | - |

### Stream 并发

```
单流执行:  [Kernel1][Kernel2][Kernel3] = 30ms
多流执行:  [Kernel1]
           [Kernel2]
           [Kernel3]                    = 12ms
```

## 🧪 运行测试

```bash
cd build

# 运行所有测试
ctest --output-on-failure

# 运行特定测试
./test_tensor
./test_memory_pool
./test_inference

# 运行 benchmark
./benchmark
./detailed_benchmark

# 运行 MNIST 演示
./mnist_demo
```

## 📚 学习要点

### 1. 内存管理

- GPU 内存分配开销大，需要池化
- 理解 cudaMalloc vs cudaMallocAsync
- 内存对齐对性能的影响

### 2. 流并发

- CUDA 流的概念和使用
- 流之间的依赖管理
- 事件同步机制

### 3. 自动调优

- Kernel 参数搜索策略
- 缓存调优结果
- 运行时配置选择

### 4. 量化推理

- FP32 → FP16 → INT8 的精度-性能权衡
- 量化校准方法
- 混合精度推理

## 🔗 相关项目

| 项目 | 关系 | 说明 |
|------|------|------|
| [01-sgemm-tutorial](../01-sgemm-tutorial) | 前置 | GEMM 优化基础 |
| [02-tensorcraft-core](../02-tensorcraft-core) | 依赖 | 提供高性能算子 |
| [03-hpc-advanced](../03-hpc-advanced) | 并行 | 进阶优化技术 |

## 📄 License

MIT License - 详见 [LICENSE](../LICENSE) 文件

---

<p align="center">
  <b>下一步</b>：尝试修改 MNIST Demo，添加新的网络层！
</p>
