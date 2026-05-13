---
outline: [2, 3]
---

# Getting Started

This document covers installation and build instructions for CUDA Kernel Academy.

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CUDA Toolkit | 12.0 | 12.x latest |
| CMake | 3.20 | 3.24+ |
| GCC / Clang | GCC 9 / Clang 10 | GCC 11+ |
| Python | 3.8 | 3.10+ |
| GPU | Volta (sm_70) | Ampere / Ada / Hopper |

## Clone Repository

```bash
git clone https://github.com/LessUp/cuda-kernel-academy.git
cd cuda-kernel-academy
```

## Build with CMake Presets

```bash
cmake --list-presets
cmake --preset default
cmake --build --preset default
ctest --preset default
```

## Build Individual Modules

### 01-sgemm-tutorial

```bash
cd 01-sgemm-tutorial
make GPU_ARCH=sm_86
./build/sgemm_benchmark
```

### 02-tensorcraft-core

```bash
cd 02-tensorcraft-core
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 03-hpc-advanced

```bash
cd 03-hpc-advanced
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 04-inference-engine

```bash
cd 04-inference-engine
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Local Quality Checks

```bash
pre-commit run --all-files
npm run docs:build
cmake --list-presets
```

## References

[^1]: NVIDIA. "CUDA C++ Programming Guide," v12.6. 2024. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
