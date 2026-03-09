# CUDA Kernel Academy

[![CI](https://github.com/LessUp/cuda-kernel-academy/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/cuda-kernel-academy/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/LessUp/cuda-kernel-academy/pages.yml?branch=main&logo=githubpages&logoColor=white&label=Docs)](https://lessup.github.io/cuda-kernel-academy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17%2F20-00599C?logo=c%2B%2B&logoColor=white)

English | [简体中文](README.zh-CN.md)

A structured CUDA programming course: from SGEMM optimization to production inference engines. Four progressive sub-projects covering GPU kernel development from basics to advanced.

## Sub-Projects

| # | Project | Focus | Tech |
|---|---------|-------|------|
| 01 | [SGEMM Tutorial](./01-sgemm-tutorial/) | Matrix multiplication optimization | CUDA C++, Makefile |
| 02 | [TensorCraft Core](./02-tensorcraft-core/) | Header-only kernel library | C++17/20, CMake |
| 03 | [HPC Advanced](./03-hpc-advanced/) | Advanced HPC techniques | CUDA, CMake, Benchmark |
| 04 | [Inference Engine](./04-inference-engine/) | DL inference engine | CUDA, CMake, pybind11 |

## Learning Path

```
01-SGEMM Tutorial (Basics)
    ↓
02-TensorCraft Core (Library Design)
    ↓
03-HPC Advanced (Optimization)
    ↓
04-Inference Engine (Application)
```

## Quick Start

```bash
git clone https://github.com/LessUp/cuda-kernel-academy.git
cd cuda-kernel-academy

# Build all sub-projects
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run tests
cd build && ctest --output-on-failure
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TENSORCRAFT` | ON | Build TensorCraft Core |
| `BUILD_HPC_ADVANCED` | ON | Build HPC Advanced |
| `BUILD_INFERENCE_ENGINE` | ON | Build Inference Engine |

## Key Topics

- **GEMM Optimization**: Naive → Tiled → Register Blocked → Tensor Core
- **Memory Hierarchy**: Global → Shared → Register, bank conflict avoidance
- **Parallel Patterns**: Reduction, scan, histogram, sort
- **Kernel Fusion**: Bias+Activation, LayerNorm+Residual
- **Mixed Precision**: FP16/BF16 Tensor Core, INT8 quantization

## Requirements

- CUDA Toolkit 12.x+
- CMake 3.20+
- C++17/20 compiler
- GPU: Volta (SM 7.0) or newer

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [Simon Boehm's CUDA Tutorials](https://siboehm.com/articles/22/CUDA-MMM)

## License

MIT License
