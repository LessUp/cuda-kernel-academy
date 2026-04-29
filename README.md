# CUDA Kernel Academy

<p align="center">
  <strong>Systematic CUDA kernel engineering from SGEMM fundamentals to reusable inference components</strong><br>
  <strong>从 SGEMM 基础到可复用推理组件的系统性 CUDA 算子工程学习路径</strong>
</p>

<p align="center">
  <a href="https://github.com/LessUp/cuda-kernel-academy/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/LessUp/cuda-kernel-academy/ci.yml?style=flat-square&logo=github&label=CI" alt="CI"></a>
  <a href="https://lessup.github.io/cuda-kernel-academy/"><img src="https://img.shields.io/github/actions/workflow/status/LessUp/cuda-kernel-academy/pages.yml?branch=master&style=flat-square&logo=githubpages&logoColor=white&label=Pages" alt="Pages"></a>
  <a href="https://github.com/LessUp/cuda-kernel-academy/releases"><img src="https://img.shields.io/github/v/release/LessUp/cuda-kernel-academy?style=flat-square&logo=github" alt="Release"></a>
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/C++-17%2F20-00599C?style=flat-square&logo=cplusplus" alt="C++">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

<div class="showcase-hero">
  <p class="showcase-kicker">A repository for people who want to understand how CUDA kernels evolve from a first correct GEMM into reusable kernels, advanced optimization experiments, and lightweight inference plumbing.</p>
  <div class="showcase-cta">
    <a href="https://lessup.github.io/cuda-kernel-academy/">Visit GitHub Pages</a>
    <a href="docs/README.md">Read the docs</a>
    <a href="01-sgemm-tutorial/README.md">Start with module 01</a>
  </div>
</div>

<div class="showcase-metrics">
  <div><strong>4</strong><span>core modules</span></div>
  <div><strong>2</strong><span>build systems</span></div>
  <div><strong>1</strong><span>OpenSpec workflow</span></div>
</div>

## Why this repo exists

Most CUDA learning material is either too small to feel like engineering or too large to understand end to end. CUDA Kernel Academy sits in the middle:

- **module 01** teaches the optimization ladder directly on SGEMM
- **module 02** turns those ideas into a reusable kernel library shape
- **module 03** explores more advanced CUDA and HPC patterns
- **module 04** shows how kernels, memory, streams, and configuration fit into a small inference-oriented system

## Project map

| Module | What you learn | Build path |
| --- | --- | --- |
| [01-sgemm-tutorial](01-sgemm-tutorial/README.md) | tiled SGEMM, bank conflicts, double buffering, WMMA | standalone `Makefile` |
| [02-tensorcraft-core](02-tensorcraft-core/README.md) | reusable kernel APIs, header-only layout, operator surface | root/module CMake |
| [03-hpc-advanced](03-hpc-advanced/README.md) | advanced optimization topics, experiments, CUDA 12+ features | root/module CMake |
| [04-inference-engine](04-inference-engine/README.md) | tensor plumbing, memory pools, streams, lightweight inference flow | root/module CMake |

## Start here

| If you want to... | Go to... |
| --- | --- |
| understand CUDA optimization from first principles | [01-sgemm-tutorial](01-sgemm-tutorial/README.md) |
| inspect a reusable kernel library layout | [02-tensorcraft-core](02-tensorcraft-core/README.md) |
| study advanced CUDA/HPC experiments | [03-hpc-advanced](03-hpc-advanced/README.md) |
| see kernels embedded in a tiny system | [04-inference-engine](04-inference-engine/README.md) |
| understand how to build, verify, and contribute | [docs/README.md](docs/README.md) |

## Quick start

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

## Build reality

- the root CMake graph covers `02-tensorcraft-core`, `03-hpc-advanced`, `04-inference-engine`, `common`, and `examples`
- `01-sgemm-tutorial` intentionally stays outside that graph
- GitHub Actions only runs CPU-safe checks
- real CUDA build and runtime validation should happen on a local GPU machine

## Documentation

- [Documentation index](docs/README.md)
- [Development workflow](docs/DEVELOPMENT_WORKFLOW.md)
- [AI tooling guide](docs/AI_TOOLING.md)
- [Installation guide](docs/INSTALLATION.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Contributing](CONTRIBUTING.md)

## Requirements

| Component | Minimum | Recommended |
| --- | --- | --- |
| CUDA Toolkit | 12.0 | 12.x |
| CMake | 3.20 | 3.24+ |
| Compiler | GCC 9 / Clang 10 | GCC 11+ |
| GPU | Volta (sm_70) | Ampere / Ada / Hopper |

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [Simon Boehm's GEMM Tutorial](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)

## Citation

```text
@misc{cuda-kernel-academy,
  author = {CUDA Kernel Academy Contributors},
  title = {CUDA Kernel Academy},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/LessUp/cuda-kernel-academy}
}
```

## License

MIT
