# CUDA Kernel Academy

[![CI](https://github.com/LessUp/cuda-kernel-academy/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/cuda-kernel-academy/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/LessUp/cuda-kernel-academy/pages.yml?branch=master&logo=githubpages&logoColor=white&label=Docs)](https://lessup.github.io/cuda-kernel-academy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17%2F20-00599C?logo=c%2B%2B&logoColor=white)

English | [简体中文](README.zh-CN.md)

A structured CUDA learning repository covering matrix multiplication, reusable kernels, advanced optimization techniques, and a lightweight inference engine.

## Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Coding Style](docs/CODING_STYLE.md)
- [Integration Examples](docs/integration_examples.md)
- [FAQ](docs/FAQ.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Contributing](CONTRIBUTING.md)

## Sub-projects

| # | Project | Focus | Build |
|---|---------|-------|-------|
| 01 | [SGEMM Tutorial](./01-sgemm-tutorial/) | Progressive SGEMM optimization | Standalone Makefile |
| 02 | [TensorCraft Core](./02-tensorcraft-core/) | Header-only kernel library | CMake |
| 03 | [HPC Advanced](./03-hpc-advanced/) | Advanced CUDA/HPC techniques | CMake |
| 04 | [Inference Engine](./04-inference-engine/) | Lightweight DL inference engine | CMake |

## Learning path

```text
01-SGEMM Tutorial
    ↓
02-TensorCraft Core
    ↓
03-HPC Advanced
    ↓
04-Inference Engine
```

## Quick start

```bash
git clone https://github.com/LessUp/cuda-kernel-academy.git
cd cuda-kernel-academy

cmake --preset default
cmake --build --preset default
ctest --preset default
```

List available presets with:

```bash
cmake --list-presets
```

## Build notes

- The root CMake build covers `02-tensorcraft-core`, `03-hpc-advanced`, `04-inference-engine`, `common`, and `examples`.
- `01-sgemm-tutorial` is intentionally standalone and uses its own `Makefile`.
- GitHub Actions currently runs CPU-safe checks (formatting, docs, links, preset validation). Full CUDA builds/tests should be run on a local machine with a GPU.

## Build options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TENSORCRAFT` | ON | Build TensorCraft Core |
| `BUILD_HPC_ADVANCED` | ON | Build HPC Advanced |
| `BUILD_INFERENCE_ENGINE` | ON | Build Inference Engine |
| `BUILD_EXAMPLES` | ON | Build examples |
| `BUILD_TESTS` | ON | Build tests |
| `BUILD_BENCHMARKS` | ON | Build benchmarks |
| `BUILD_PYTHON_BINDINGS` | OFF | Build optional Python bindings |

## Requirements

- CUDA Toolkit 12.x+
- CMake 3.20+
- C++17/20 compiler
- NVIDIA GPU with SM 7.0+

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [Simon Boehm's CUDA Tutorials](https://siboehm.com/articles/22/CUDA-MMM)

## License

MIT License
