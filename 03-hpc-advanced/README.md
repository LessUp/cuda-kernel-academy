# HPC Advanced

进阶 CUDA / HPC 教程模块，聚焦更复杂的算子组织方式、GEMM 优化路径、属性测试和新架构相关实验。

## 模块定位

`03-hpc-advanced` 面向已经完成 `01-sgemm-tutorial` 和 `02-tensorcraft-core` 的读者。

它包含：

- 更完整的 GEMM 优化路径
- reduction / attention / convolution / quantization 示例
- GoogleTest + RapidCheck 测试
- 可选 Python 绑定
- `07_cuda13_features/` 中的 CUDA 12/13 相关实验性实现

## 重要说明

`src/07_cuda13_features/` 当前主要是**教学性质的占位/兼容实现**，用于展示接口、组织方式和实验方向；它们**不应被视为完整的 Hopper / Blackwell 生产级实现**。

尤其是：

- TMA
- Thread Block Clusters
- FP8 GEMM

这些内容更适合作为学习入口和后续扩展基础，而不是对真实硬件特性的完整封装。

## 目录结构

```text
03-hpc-advanced/
├── src/
│   ├── common/
│   ├── 01_elementwise/
│   ├── 02_reduction/
│   ├── 03_gemm/
│   ├── 04_convolution/
│   ├── 05_attention/
│   ├── 06_quantization/
│   └── 07_cuda13_features/
├── python/
├── tests/
├── docker/
└── CMakeLists.txt
```

## 环境要求

- CUDA 12.x+ 可构建本模块主体
- CMake 3.20+
- GCC 11+ / Clang 14+（建议）
- 支持 CUDA 的 NVIDIA GPU

## 构建

```bash
cd 03-hpc-advanced
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

## 内容概览

### GEMM 优化路径

- Naive
- Shared Memory Tiling
- Double Buffering
- Register Tiling
- WMMA / Tensor Core
- MMA PTX
- Software Pipelining

### 其他算子

- Elementwise
- Reduction
- LayerNorm / Softmax
- FlashAttention / RoPE / TopK
- Convolution
- Quantization

## 属性测试

本模块集成了 RapidCheck，用于补充传统单元测试。

## Docker

`docker/` 目录提供了便于本地实验的开发环境示例，但不应替代在真实 GPU 机器上的最终验证。

## 相关模块

- [01-sgemm-tutorial](../01-sgemm-tutorial)
- [02-tensorcraft-core](../02-tensorcraft-core)
- [04-inference-engine](../04-inference-engine)
