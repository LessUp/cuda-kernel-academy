# CUDA Kernel Academy

> GitHub repository: [`cuda-foundations`](https://github.com/AICL-Lab/cuda-foundations)
> （教学品牌名：CUDA Kernel Academy）
>
> 📚 Portfolio map: https://github.com/aicl-lab/aicl-lab

<p align="center">
  <strong>Systematic CUDA kernel engineering from SGEMM fundamentals to reusable inference components</strong><br>
  <strong>从 SGEMM 基础到可复用推理组件的系统性 CUDA 算子工程学习路径</strong>
</p>

<p align="center">
  <a href="https://github.com/AICL-Lab/cuda-foundations/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/AICL-Lab/cuda-foundations/ci.yml?style=flat-square&logo=github&label=CI" alt="CI"></a>
  <a href="https://aicl-lab.github.io/cuda-foundations/"><img src="https://img.shields.io/github/actions/workflow/status/AICL-Lab/cuda-foundations/pages.yml?branch=master&style=flat-square&logo=githubpages&logoColor=white&label=Pages" alt="Pages"></a>
  <a href="https://github.com/AICL-Lab/cuda-foundations/releases"><img src="https://img.shields.io/github/v/release/AICL-Lab/cuda-foundations?style=flat-square&logo=github" alt="Release"></a>
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
    <a href="https://aicl-lab.github.io/cuda-foundations/">Visit GitHub Pages</a>
    <a href="docs/en/index.md">Read the docs</a>
    <a href="01-sgemm-tutorial/README.md">Start with module 01</a>
  </div>
</div>

<div class="showcase-metrics">
  <div><strong>4</strong><span>core modules</span></div>
  <div><strong>2</strong><span>build systems</span></div>
  <div><strong>1</strong><span>root changelog</span></div>
</div>

## Why this repo exists

Most CUDA learning material is either too small to feel like engineering or too large to understand end to end. CUDA Kernel Academy sits in the middle:

- **module 01** teaches the optimization ladder directly on SGEMM
- **module 02** turns those ideas into a reusable kernel library shape
- **module 03** explores more advanced CUDA and HPC patterns
- **module 04** shows how kernels, memory, streams, and configuration fit into a small inference-oriented system

## Repository status

Teaching repository, **Phase 2 interview-ready freeze** (`phase-2-e`).
The four modules are complete. On the 2026-08-18 freeze machine
`ctest --preset default` reported **0 failed / 209 collected**
(131 ran, 78 skipped — mostly 02/04 GPU binaries; same skip class as
the prior freeze note). No new modules will be added here. Follow-up
work lives in [LEARNING_PATH.md](LEARNING_PATH.md) (e.g. `tiny-llm`
and `paged-infer`).

## Consolidation boundary

CUDA Kernel Academy is the maintained successor for the overlapping learning paths that previously lived in `sgemm-optimization`, `modern-ai-kernels`, `hpc-ai-optimization-lab`, and `mini-inference-engine`. Useful implementations and verification now live in the matching Academy modules; the duplicate repositories have been retired instead of remaining as parallel maintenance surfaces.

The former `llm-speed` GEMM learning path is covered here, while the maintained deep FlashAttention implementation lives in [`cuflash-attn`](https://github.com/AICL-Lab/cuflash-attn). The Academy intentionally keeps only one implementation per teaching role instead of importing every historical variant.

## Scope（IN / OUT）

**IN（本仓库负责）**：
- CUDA 编程模型与 kernel 编写（线程/块/共享内存/同步）
- GEMM 优化阶梯（naive → tiled → bank-conflict-free → double buffering → WMMA）
- 算子库设计（header-only 布局、operator surface）
- profiling 与性能分析方法

**OUT（明确不做，见对应仓库）**：
- 生产级推理运行时（模型加载/采样/生成）→ [tiny-llm](https://github.com/AICL-Lab/tiny-llm)
- 完整 FlashAttention 前后向与优化 → [cuflash-attn](https://github.com/AICL-Lab/cuflash-attn)
- Serving / 分页 KV / continuous batching → [paged-infer](https://github.com/AICL-Lab/paged-infer)
- 本仓库 [04-inference-engine](04-inference-engine/README.md) 为**教学预览**，非独立作品

## Project map

| Module | What you learn | Build path |
| --- | --- | --- |
| [01-sgemm-tutorial](01-sgemm-tutorial/README.md) | tiled SGEMM, bank conflicts, double buffering, WMMA | root CMake + optional `Makefile` |
| [02-tensorcraft-core](02-tensorcraft-core/README.md) | reusable kernel APIs, header-only layout, operator surface | root/module CMake |
| [03-hpc-advanced](03-hpc-advanced/README.md) | advanced optimization topics, experiments, CUDA 12+ features | root/module CMake |
| [04-inference-engine](04-inference-engine/README.md)（教学预览） | tensor plumbing, memory pools, streams, lightweight inference flow | root/module CMake |

## Start here

| If you want to... | Go to... |
| --- | --- |
| understand CUDA optimization from first principles | [01-sgemm-tutorial](01-sgemm-tutorial/README.md) |
| inspect a reusable kernel library layout | [02-tensorcraft-core](02-tensorcraft-core/README.md) |
| study advanced CUDA/HPC experiments | [03-hpc-advanced](03-hpc-advanced/README.md) |
| see kernels embedded in a tiny system | [04-inference-engine](04-inference-engine/README.md) |
| understand how to build, verify, and contribute | [docs/en/index.md](docs/en/index.md) |

For the repository sequence from CUDA fundamentals to LLM serving, see the Chinese [AI Infra beginner learning path](LEARNING_PATH.md). It also records the shared optimization and invariant-testing method used across the five retained projects.

## Quick start

```bash
git clone https://github.com/AICL-Lab/cuda-foundations.git
cd cuda-foundations

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

- the root CMake graph covers `01-sgemm-tutorial`, `02-tensorcraft-core`, `03-hpc-advanced`, `04-inference-engine`, `common`, and `examples`
- the `01-sgemm-tutorial/Makefile` remains available for quick standalone runs, but tests and CI use the root CMake graph
- GitHub Actions only runs CPU-safe checks
- real CUDA build and runtime validation should happen on a local GPU machine

## Documentation

- [Documentation index](docs/en/index.md)
- [Getting started](docs/en/guides/getting-started.md)
- [Development workflow](docs/en/guides/workflow.md)
- [Coding style](docs/en/reference/coding-style.md)
- [Troubleshooting](docs/en/reference/troubleshooting.md)
- [Changelog](CHANGELOG.md)
- [中文文档首页](docs/zh/index.md)
- [Follow-up development plan (executable by coding agents)](DEV_PLAN.md)
- [Contributing](CONTRIBUTING.md)
- 📖 [技术文档站](https://aicl-lab.github.io/cuda-foundations/) 由 VitePress 驱动，支持 NVIDIA 风格深色主题、交互式图表和学术论文引用。

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
@misc{cuda-foundations,
  author = {CUDA Kernel Academy Contributors},
  title = {CUDA Kernel Academy},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/AICL-Lab/cuda-foundations}
}
```

## License

MIT
