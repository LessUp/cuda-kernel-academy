---
outline: [2, 3]
---

# CUDA Kernel Academy

Systematic CUDA kernel engineering from SGEMM fundamentals to reusable inference components.

## Project Overview

CUDA Kernel Academy is a multi-module CUDA learning repository with four core modules:

| Module | What You Learn | Build System |
|--------|----------------|--------------|
| [01-SGEMM Tutorial](./modules/01-sgemm.md) | Tiled SGEMM, bank conflicts, double buffering, WMMA | Standalone Makefile |
| [02-TensorCraft Core](./modules/02-tensorcraft.md) | Reusable kernel APIs, header-only layout | Root/module CMake |
| [03-HPC Advanced](./modules/03-hpc.md) | Advanced optimization, CUDA 12+ features | Root/module CMake |
| [04-Inference Engine](./modules/04-inference.md) | Tensor plumbing, memory pools, streams | Root/module CMake |

## Quick Start

```bash
git clone https://github.com/LessUp/cuda-kernel-academy.git
cd cuda-kernel-academy

cmake --list-presets
cmake --preset default
cmake --build --preset default
ctest --preset default
```

For the standalone tutorial:

```bash
cd 01-sgemm-tutorial
make GPU_ARCH=sm_86
make test
```

## Documentation

- [Getting Started](./guides/getting-started.md)
- [Development Workflow](./guides/workflow.md)
- [Benchmarks](./benchmarks/)
- [Roadmap](./roadmap.md)
- [Coding Style](./reference/coding-style.md)
- [Troubleshooting](./reference/troubleshooting.md)

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CUDA Toolkit | 12.0 | 12.x |
| CMake | 3.20 | 3.24+ |
| Compiler | GCC 9 / Clang 10 | GCC 11+ |
| GPU | Volta (sm_70) | Ampere / Ada / Hopper |
