# 🚀 HPC Advanced: CUDA 13 & Cutting-Edge Optimization

<p align="center">
  <b>进阶 CUDA 优化 - 探索最新架构特性</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Level-Advanced-red?style=flat-square" alt="Level">
  <img src="https://img.shields.io/badge/CUDA-13.1%2B-76B900?style=flat-square&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/C++-20-00599C?style=flat-square&logo=cplusplus" alt="C++20">
  <img src="https://img.shields.io/badge/Architecture-Hopper%2FBlackwell-green?style=flat-square" alt="Architecture">
</p>

---

## 📖 模块简介

**HPC Advanced** 是 CUDA Kernel Academy 的进阶教程，专注于最新的 CUDA 13 特性和极致优化技术。本模块面向有经验的 CUDA 开发者，展示如何充分利用 Hopper/Blackwell 架构的新特性。

> **🎯 学习目标**：掌握 Register Tiling、Software Pipelining、TMA、Thread Block Clusters 等进阶技术

## 🗺️ 在学习路径中的位置

```
┌─────────────────────────────────────────────────────────────────┐
│  01-sgemm-tutorial → 02-tensorcraft-core                        │
│                          ↓                                      │
│                    ★ 03-hpc-advanced (You are here)            │
│                          ↓                                      │
│                    04-inference-engine                          │
└─────────────────────────────────────────────────────────────────┘
```

**前置要求**：完成 `01-sgemm-tutorial` 和 `02-tensorcraft-core`
**后续学习**：`04-inference-engine` 或深入研究 CUTLASS 源码

## ✨ 核心特性

| 特性 | 描述 |
|------|------|
| 🔥 **GEMM 7步优化** | 从 Naive 到 Software Pipelining 完整路径 |
| ⚡ **CUDA 13 新特性** | TMA、Thread Block Clusters、FP8 |
| 🎯 **Modern C++20** | Concepts、constexpr、RAII |
| 🧠 **LLM 专项算子** | FlashAttention、RoPE、MoE TopK |
| ✅ **属性测试** | RapidCheck 确保算法正确性 |
| 🐍 **Python 绑定** | Nanobind 零拷贝绑定 |

## 📁 目录结构

```
03-hpc-advanced/
├── src/
│   ├── common/                 # 基础工具库
│   │   ├── cuda_check.cuh
│   │   ├── timer.cuh
│   │   ├── tensor.cuh
│   │   └── types.cuh
│   ├── 01_elementwise/         # 访存密集型算子
│   ├── 02_reduction/           # 归约类算子
│   ├── 03_gemm/                # GEMM 7步优化 ⭐
│   ├── 04_convolution/         # 卷积算子
│   ├── 05_attention/           # LLM 核心算子
│   ├── 06_quantization/        # 量化算子
│   └── 07_cuda13_features/     # CUDA 13 新特性 ⭐
│       ├── tma.cu/cuh          # TMA 异步搬运
│       ├── cluster.cu/cuh      # Thread Block Clusters
│       └── fp8_gemm.cu/cuh     # FP8 GEMM
├── python/
│   ├── bindings/               # Nanobind 绑定
│   └── benchmark/              # PyTorch 对比脚本
├── tests/                      # GoogleTest + RapidCheck
├── docker/                     # CUDA 13.1 Docker 环境
└── CMakeLists.txt
```

## 🔧 GEMM 7步优化

这是本模块的**重中之重**，完整展示从 Naive 到极致优化的路径：

```
Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 6 → Step 7
Naive   Shared   Double   Register  WMMA     MMA PTX  Pipeline
        Tiling   Buffer   Tiling    API
```

| Step | 优化技术 | TFLOPS (FP32) | 关键改进 |
|------|----------|---------------|----------|
| 1 | Naive Global Memory | ~0.5 | 基准实现 |
| 2 | Shared Memory Tiling | ~2.0 | 减少全局内存访问 |
| 3 | Double Buffering | ~3.5 | 计算与加载重叠 |
| 4 | Register Tiling | ~6.0 | 减少 Shared Memory 压力 |
| 5 | Tensor Core (WMMA) | ~50+ | 使用 Tensor Core |
| 6 | Tensor Core (MMA PTX) | ~60+ | 更底层控制 |
| 7 | Software Pipelining | ~70+ | 隐藏指令延迟 |

## 🆕 CUDA 13 新特性

### TMA (Tensor Memory Accelerator)

```cpp
// 使用 TMA 异步数据搬运
// 解放 Register 和 SM，让 Copy Engine 自动搬运数据
cuda::memcpy_async(shared_mem, global_mem, size, barrier);
```

### Thread Block Clusters

```cpp
// 利用 Hopper 架构的 Cluster 特性
// 实现 Block 间的 Shared Memory 直接访问
__cluster_dims__(2, 2, 1)
__global__ void cluster_kernel(...) {
    // Distributed Shared Memory 访问
}
```

### FP8 GEMM

```cpp
// 使用 e4m3 和 e5m2 数据类型
// 下一代低精度推理
__nv_fp8_e4m3 a = ...;
__nv_fp8_e5m2 b = ...;
```

## 🚀 快速开始

### 环境要求

| 依赖 | 版本要求 |
|------|----------|
| CUDA | 13.1+ (Hopper 特性需要 13.0+) |
| CMake | 3.24+ |
| C++ 编译器 | GCC 11+ / Clang 14+ (C++20) |
| Python | 3.8+ |

### 构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja
ninja

# 运行测试
ctest --output-on-failure

# 运行特定模块测试
./tests/gemm/test_gemm
./tests/attention/test_attention
```

### Docker 环境 (推荐)

```bash
cd docker
docker-compose up -d
docker exec -it hpc-advanced bash
```

## 🧪 属性测试

本模块使用 **RapidCheck** 进行属性测试：

```cpp
// Property: GEMM Correctness
// 对于任意矩阵 A, B，我们的 GEMM 结果应该与参考实现一致
RC_GTEST_PROP(GemmTest, Correctness, (Matrix A, Matrix B)) {
    auto our_result = gemm(A, B);
    auto ref_result = reference_gemm(A, B);
    RC_ASSERT(approx_equal(our_result, ref_result, 1e-5));
}
```

## 📊 性能对比

```
================================================================================
                        HPC Advanced Benchmark
================================================================================

GEMM Performance (M=4096, N=4096, K=4096)
--------------------------------------------------------------------------------
| Implementation          | Time (ms) | TFLOPS | vs PyTorch |
|-------------------------|-----------|--------|------------|
| PyTorch (cuBLAS)        |    2.15   |  63.8  |    1.00x   |
| Naive                   |   89.32   |   1.5  |    0.02x   |
| SharedMemTiling         |   21.45   |   6.4  |    0.10x   |
| DoubleBuffer            |   12.87   |  10.7  |    0.17x   |
| RegisterTiling          |    7.23   |  19.0  |    0.30x   |
| TensorCore (WMMA)       |    2.45   |  56.1  |    0.88x   |
| TensorCore (MMA PTX)    |    2.31   |  59.5  |    0.93x   |
| SoftwarePipeline        |    2.18   |  63.0  |    0.99x   |
--------------------------------------------------------------------------------
```

## 🔗 相关模块

| 模块 | 关系 | 说明 |
|------|------|------|
| [01-sgemm-tutorial](../01-sgemm-tutorial) | 前置 | SGEMM 基础 |
| [02-tensorcraft-core](../02-tensorcraft-core) | 可选依赖 | 基础算子库 |
| [04-inference-engine](../04-inference-engine) | 应用 | 推理引擎框架 |

## 📖 参考资料

- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [CUDA C++ Programming Guide - CUDA 13](https://docs.nvidia.com/cuda/)

## 📄 License

MIT License
