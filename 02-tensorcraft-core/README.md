# 🔧 TensorCraft Core: High-Performance AI Kernel Library

<p align="center">
  <b>工业级高性能 AI 算子库 - Header-Only 设计</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Level-Intermediate-yellow?style=flat-square" alt="Level">
  <img src="https://img.shields.io/badge/CUDA-11.0%2B-76B900?style=flat-square&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/C++-17%2F20-00599C?style=flat-square&logo=cplusplus" alt="C++">
  <img src="https://img.shields.io/badge/Header--Only-Yes-brightgreen?style=flat-square" alt="Header-Only">
</p>

---

## 📖 模块简介

**TensorCraft Core** 是 CUDA Kernel Academy 的核心算子库，提供生产级的 CUDA kernel 实现。采用 Header-Only 设计，易于集成到其他项目中。

> **🎯 学习目标**：掌握工业级 API 设计、理解多种算子实现、学会在项目中集成高性能库

## 🗺️ 在学习路径中的位置

```
┌─────────────────────────────────────────────────────────────────┐
│  01-sgemm-tutorial                                              │
│     ↓                                                           │
│  ★ 02-tensorcraft-core (You are here)                          │
│     ↓                                                           │
│  03-hpc-advanced / 04-inference-engine                         │
└─────────────────────────────────────────────────────────────────┘
```

**前置要求**：完成 `01-sgemm-tutorial`
**后续学习**：`03-hpc-advanced` (进阶) 或 `04-inference-engine` (应用)

## ✨ 核心特性

| 特性 | 描述 |
|------|------|
| 📦 **Header-Only** | 纯头文件设计，零编译依赖 |
| 🚀 **多架构支持** | Volta (sm_70) 到 Hopper (sm_90) |
| 🐍 **Python 绑定** | pybind11 提供 Python 接口 |
| 🧪 **完整测试** | GoogleTest 单元测试 |
| 📊 **性能基准** | Google Benchmark 性能测试 |

## 📁 目录结构

```
02-tensorcraft-core/
├── include/tensorcraft/
│   ├── core/
│   │   ├── cuda_check.hpp      # CUDA 错误检查 (统一标准)
│   │   ├── features.hpp        # 特性检测
│   │   └── type_traits.hpp     # 类型特征
│   ├── kernels/
│   │   ├── gemm.hpp            # GEMM 实现
│   │   ├── attention.hpp       # Attention 机制
│   │   ├── normalization.hpp   # LayerNorm / RMSNorm
│   │   ├── elementwise.hpp     # 激活函数
│   │   ├── softmax.hpp         # Softmax
│   │   ├── conv2d.hpp          # 卷积
│   │   ├── sparse.hpp          # 稀疏矩阵
│   │   └── fusion.hpp          # 算子融合
│   └── memory/
│       ├── tensor.hpp          # Tensor 封装
│       └── memory_pool.hpp     # 内存池
├── src/python_ops/             # Python 绑定
├── tests/                      # 单元测试
├── benchmarks/                 # 性能基准
├── docs/                       # 文档
└── CMakeLists.txt
```

## 🚀 快速开始

### 环境要求

- CMake 3.20+
- CUDA Toolkit 11.0+
- C++17 兼容编译器

### 构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 运行测试
ctest --output-on-failure

# 运行 benchmark
./benchmarks/gemm_benchmark
```

### 在项目中使用

**方式 1: 作为子目录**

```cmake
add_subdirectory(02-tensorcraft-core)
target_link_libraries(your_target PRIVATE tensorcraft)
```

**方式 2: 直接包含头文件**

```cmake
target_include_directories(your_target PRIVATE 
    path/to/02-tensorcraft-core/include)
target_link_libraries(your_target PRIVATE CUDA::cudart)
```

## 📚 API 使用示例

### GEMM 矩阵乘法

```cpp
#include "tensorcraft/kernels/gemm.hpp"

using namespace tensorcraft::kernels;

// 选择优化版本
launch_gemm(d_A, d_B, d_C, M, N, K, 
            1.0f, 0.0f,                    // alpha, beta
            GemmVersion::Tiled,            // 版本选择
            stream);

// Tensor Core (half 精度)
#ifdef TC_HAS_WMMA
launch_gemm_wmma(d_A_half, d_B_half, d_C_float, M, N, K);
#endif
```

### 归一化层

```cpp
#include "tensorcraft/kernels/normalization.hpp"

// LayerNorm
layernorm(d_input, d_gamma, d_beta, d_output, batch_size, hidden_size);

// RMSNorm (LLaMA, Mistral 使用)
rmsnorm(d_input, d_weight, d_output, batch_size, hidden_size);
```

### Attention 机制

```cpp
#include "tensorcraft/kernels/attention.hpp"

// FlashAttention 风格
float scale = 1.0f / sqrtf(head_dim);
launch_flash_attention(d_Q, d_K, d_V, d_O,
                       batch_size, num_heads, seq_len, head_dim, scale);

// RoPE 位置编码
launch_rope(d_x, d_cos, d_sin, batch_size, seq_len, num_heads, head_dim);
```

### Python 接口

```python
import tensorcraft_ops as tc
import numpy as np

# GEMM
A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(512, 128).astype(np.float32)
C = tc.gemm(A, B, version='tiled')

# 激活函数
output = tc.relu(input_data)
output = tc.gelu(input_data)

# 归一化
output = tc.layernorm(input_data, gamma, beta)
```

## 🔧 统一 API 设计

### CUDA 错误检查 (标准实现)

```cpp
#include "tensorcraft/core/cuda_check.hpp"

// API 调用检查
TC_CUDA_CHECK(cudaMalloc(&ptr, size));

// Kernel 启动检查
my_kernel<<<grid, block>>>(...);
TC_CUDA_CHECK_LAST();

// 同步检查
TC_CUDA_SYNC_CHECK();
```

### Kernel 版本枚举

```cpp
enum class GemmVersion {
    Naive,          // 基础实现
    Tiled,          // 共享内存分块
    DoubleBuffer,   // 双缓冲
    TensorCore,     // WMMA Tensor Core
    Auto            // 自动选择
};
```

## 📊 性能基准

在 NVIDIA RTX 3090 上的 GEMM 性能：

| 矩阵大小 | Naive | Tiled | Double Buffer | cuBLAS |
|----------|-------|-------|---------------|--------|
| 512×512 | 15 GFLOPS | 180 GFLOPS | 220 GFLOPS | 280 GFLOPS |
| 1024×1024 | 18 GFLOPS | 350 GFLOPS | 450 GFLOPS | 520 GFLOPS |
| 2048×2048 | 20 GFLOPS | 480 GFLOPS | 620 GFLOPS | 750 GFLOPS |

## 🔗 相关模块

| 模块 | 关系 | 说明 |
|------|------|------|
| [01-sgemm-tutorial](../01-sgemm-tutorial) | 前置 | SGEMM 基础优化 |
| [03-hpc-advanced](../03-hpc-advanced) | 扩展 | 进阶优化技术 |
| [04-inference-engine](../04-inference-engine) | 依赖方 | 使用本库构建推理引擎 |

## 📄 License

MIT License
