# CUDA Kernel Academy

<p align="center">
  <b>From Zero to Hero: Systematic CUDA High-Performance Kernel Development</b><br>
  <b>从零到极致：系统性学习 CUDA 高性能算子开发</b>
</p>

<p align="center">
  <a href="https://github.com/LessUp/cuda-kernel-academy/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/LessUp/cuda-kernel-academy/ci.yml?style=flat-square&logo=github&label=CI" alt="CI">
  </a>
  <a href="https://lessup.github.io/cuda-kernel-academy/">
    <img src="https://img.shields.io/github/actions/workflow/status/LessUp/cuda-kernel-academy/pages.yml?branch=master&style=flat-square&logo=githubpages&logoColor=white&label=Docs" alt="Docs">
  </a>
  <a href="https://github.com/LessUp/cuda-kernel-academy/releases">
    <img src="https://img.shields.io/github/v/release/LessUp/cuda-kernel-academy?style=flat-square&logo=github" alt="Release">
  </a>
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/C++-17%2F20-00599C?style=flat-square&logo=cplusplus" alt="C++">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License">
  </a>
  <a href="https://github.com/LessUp/cuda-kernel-academy/stargazers">
    <img src="https://img.shields.io/github/stars/LessUp/cuda-kernel-academy?style=flat-square&logo=github" alt="Stars">
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

---

A structured CUDA learning repository covering matrix multiplication, reusable kernels, advanced optimization techniques, and a lightweight inference engine. Master GPU programming from SGEMM basics to Tensor Core optimization.

## Features

| Feature | Description |
|---------|-------------|
| **Progressive Learning Path** | 4 interconnected modules from basics to production |
| **Performance-Focused** | Real benchmarks against cuBLAS, not toy examples |
| **Modern C++** | Leverages C++17/20 features for clean, safe GPU code |
| **Production Patterns** | Header-only library design, memory pools, stream management |
| **Multi-Architecture** | Supports Volta (sm_70) through Hopper (sm_90) |

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
01-SGEMM Tutorial (1-2 weeks)
        ↓  Master shared memory, bank conflicts, WMMA
02-TensorCraft Core (2-3 weeks)
        ↓  Build reusable kernels, API design
03-HPC Advanced (3-4 weeks)
        ↓  CUDA 13 features, FlashAttention
04-Inference Engine (2-3 weeks)
        ↓  Complete inference framework
```

**Prerequisites**: C/C++ basics, linear algebra fundamentals. CUDA experience helpful but not required.

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

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CUDA Toolkit | 11.0 | 12.x |
| CMake | 3.20 | 3.24+ |
| Compiler | GCC 9 / Clang 10 | GCC 11+ |
| GPU | Volta (sm_70) | Ampere/Ada (sm_80+) |

**Supported Architectures**:

| Arch | sm | GPUs |
|------|-----|------|
| Volta | 70 | V100 |
| Turing | 75 | RTX 2080, T4 |
| Ampere | 80, 86 | A100, RTX 3090 |
| Ada | 89 | RTX 4090, L40 |
| Hopper | 90 | H100 |

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass) - CUDA Templates for Linear Algebra
- [Simon Boehm's GEMM Tutorial](https://siboehm.com/articles/22/CUDA-MMM) - Excellent optimization walkthrough
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/) - Latest techniques and best practices

## Citation

If you find this project helpful in your research or work:

```bibtex
@misc{cuda-kernel-academy,
  author = {CUDA Kernel Academy Contributors},
  title = {CUDA Kernel Academy: A Comprehensive Learning Path for High-Performance CUDA Kernel Development},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/LessUp/cuda-kernel-academy}
}
```

## License

MIT License
