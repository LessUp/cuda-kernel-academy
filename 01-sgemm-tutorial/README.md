# 🎓 SGEMM Tutorial: From Naive to Tensor Core

本模块是 CUDA Kernel Academy 的入门教程，聚焦于 SGEMM 的渐进式优化过程。

## 内容概览

- Naive SGEMM
- Shared-memory tiled SGEMM
- Bank-conflict-free SGEMM
- Double-buffered SGEMM
- Tensor Core / WMMA SGEMM
- 与 cuBLAS 对比的 benchmark
- 基于 GoogleTest 的正确性测试

## 目录结构

```text
01-sgemm-tutorial/
├── src/
│   ├── kernels/
│   ├── utils/
│   └── main.cu
├── tests/
│   └── test_sgemm.cu
├── Makefile
└── README.md
```

## 构建方式

该模块是**独立教程模块**，不参与根目录 CMake 构建。

### 构建 benchmark

```bash
cd 01-sgemm-tutorial
make GPU_ARCH=sm_86
./build/sgemm_benchmark
```

### 运行测试

```bash
cd 01-sgemm-tutorial
make test
```

说明：

- `make test` 依赖系统可用的 GoogleTest（例如通过包管理器安装 `gtest` / `libgtest-dev`）。
- `GPU_ARCH` 可按本机 GPU 调整，例如 `sm_80`、`sm_86`、`sm_89`、`sm_90`。

## 学习路径中的位置

```text
01-sgemm-tutorial
    ↓
02-tensorcraft-core
    ↓
03-hpc-advanced
    ↓
04-inference-engine
```

## 适合什么人

- 想理解 CUDA GEMM 优化基本路径的学习者
- 想从最小示例入手理解 shared memory、bank conflict、double buffering、WMMA 的读者

## 相关模块

- [02-tensorcraft-core](../02-tensorcraft-core)
- [03-hpc-advanced](../03-hpc-advanced)
- [04-inference-engine](../04-inference-engine)
