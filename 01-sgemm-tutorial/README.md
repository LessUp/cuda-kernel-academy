# 🎓 SGEMM Tutorial: From Naive to Tensor Core

<p align="center">
  <b>从零手写极致优化的矩阵乘法 - HPC 领域的 "Hello World"</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Level-Beginner-brightgreen?style=flat-square" alt="Level">
  <img src="https://img.shields.io/badge/CUDA-11.0%2B-76B900?style=flat-square&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/Build-Makefile-orange?style=flat-square" alt="Build">
</p>

---

## 📖 模块简介

本模块是 **CUDA Kernel Academy** 的入门教程，专注于 SGEMM (Single-precision General Matrix Multiply) 的渐进式优化。通过5种优化版本，展示 GPU 编程中的核心优化技术。

> **🎯 学习目标**：理解 CUDA 内存层次、掌握基础优化技术、达到 cuBLAS 40%+ 性能

## 🗺️ 在学习路径中的位置

```
┌─────────────────────────────────────────────────────────────────┐
│  ★ 01-sgemm-tutorial (You are here)                             │
│     ↓                                                           │
│  02-tensorcraft-core → 03-hpc-advanced → 04-inference-engine   │
└─────────────────────────────────────────────────────────────────┘
```

**前置要求**：C++ 基础、CUDA 编程入门
**后续学习**：完成本模块后，建议学习 `02-tensorcraft-core`

## 📊 性能结果

在 NVIDIA GeForce RTX 3060 Laptop GPU (sm_86) 上的 1024×1024×1024 矩阵乘法：

| Kernel | GFLOPS | vs cuBLAS | 状态 |
|--------|--------|-----------|------|
| cuBLAS (参考) | 5727 | 100% | ✅ |
| Tensor Core (WMMA) | 2300 | 40.2% | ✅ |
| Tiled (32×32) | 753 | 13.1% | ✅ |
| Double Buffer | 701 | 12.2% | ✅ |
| Bank Conflict Free | 673 | 11.8% | ✅ |
| Naive | 604 | 10.6% | ✅ |

## 🔧 优化版本

| 版本 | 描述 | 关键技术 | 文件 |
|------|------|----------|------|
| v1 | Naive | 基础三层循环 | `naive_sgemm.cuh` |
| v2 | Tiled | 共享内存分块 | `tiled_sgemm.cuh` |
| v3 | Bank-Free | 消除 bank 冲突 | `bank_conflict_free_sgemm.cuh` |
| v4 | Double Buffer | 双缓冲流水线 | `double_buffer_sgemm.cuh` |
| v5 | Tensor Core | WMMA API | `tensor_core_sgemm.cuh` |

## 📁 目录结构

```
01-sgemm-tutorial/
├── src/
│   ├── kernels/
│   │   ├── naive_sgemm.cuh           # v1: 基础实现
│   │   ├── tiled_sgemm.cuh           # v2: 共享内存分块
│   │   ├── bank_conflict_free_sgemm.cuh  # v3: 消除 bank 冲突
│   │   ├── double_buffer_sgemm.cuh   # v4: 双缓冲
│   │   └── tensor_core_sgemm.cuh     # v5: Tensor Core
│   ├── utils/
│   │   ├── cuda_utils.cuh            # CUDA 工具函数
│   │   ├── benchmark.cuh             # 性能测试框架
│   │   └── verify.cuh                # 正确性验证
│   └── main.cu                       # 主程序
├── tests/
│   └── test_sgemm.cu                 # 单元测试
├── Makefile
└── README.md
```

## 🚀 快速开始

### 环境要求

- CUDA Toolkit 11.0+
- GPU: Volta (sm_70) 或更新架构
- cuBLAS (CUDA 自带)

### 编译运行

```bash
# 编译 (根据你的 GPU 架构调整)
make GPU_ARCH=sm_86

# 运行 benchmark
./build/sgemm_benchmark

# 运行测试
make test
```

### 输出示例

```
===============================================================
   SGEMM Optimization Benchmark Suite
===============================================================
GPU Device: NVIDIA GeForce RTX 3060 Laptop GPU
Compute Capability: 8.6

  Kernel              |    Time |  Performance | Pass
-----------------------------------------------------------------------
  cuBLAS              | 0.375ms | 5726 GFLOPS  | PASS
  Naive               | 3.553ms |  604 GFLOPS  | PASS
  Tiled (32x32)       | 2.853ms |  753 GFLOPS  | PASS
  Bank Conflict Free  | 3.190ms |  673 GFLOPS  | PASS
  Double Buffer       | 3.064ms |  701 GFLOPS  | PASS
  Tensor Core (WMMA)  | 0.934ms | 2300 GFLOPS  | PASS
```

## 📚 核心概念

### 1. 内存合并访问 (Memory Coalescing)

```cpp
// ❌ 非合并访问 - 带宽浪费
float b = B[k * N + col];  // stride = N

// ✅ 合并访问 - 高效利用带宽
Bs[ty][tx] = B[bRow * N + bCol];  // stride = 1
```

### 2. 共享内存分块 (Tiling)

```cpp
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

// 数据复用: 每个元素被读取 1 次，使用 TILE_SIZE 次
for (int k = 0; k < TILE_SIZE; ++k) {
    sum += As[ty][k] * Bs[k][tx];
}
```

### 3. Bank Conflict 消除

```cpp
// ❌ 有 bank conflict
__shared__ float As[32][32];

// ✅ 无 bank conflict (+1 padding)
__shared__ float As[32][33];
```

### 4. 双缓冲流水线

```cpp
__shared__ float As[2][TILE_SIZE][TILE_SIZE];  // 双缓冲
// Load[i+1] 与 Compute[i] 重叠执行
```

### 5. Tensor Core (WMMA)

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
mma_sync(c_frag, a_frag, b_frag, c_frag);
```

## 🔗 相关模块

| 模块 | 关系 | 说明 |
|------|------|------|
| [02-tensorcraft-core](../02-tensorcraft-core) | 下一步 | 学习更多算子和工业级 API |
| [03-hpc-advanced](../03-hpc-advanced) | 进阶 | Register Tiling、CUDA 13 特性 |
| [04-inference-engine](../04-inference-engine) | 应用 | 完整推理引擎框架 |

## 📖 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
- [Roofline Model](https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/)

## 📄 License

MIT License
