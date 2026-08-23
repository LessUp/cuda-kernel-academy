# cuda-foundations

> GitHub 仓库名：[`cuda-foundations`](https://github.com/open-infra-ai/cuda-foundations)
> 教学品牌名：CUDA Kernel Academy
>
> 📚 作品集地图：<https://github.com/open-infra-ai/open-infra-ai>

<p align="center">
  <strong>从 SGEMM 基础到可复用推理组件的系统性 CUDA 算子工程学习路径</strong><br>
  <strong>Systematic CUDA kernel engineering from SGEMM fundamentals to reusable inference components</strong>
</p>

<p align="center">
  <a href="https://github.com/open-infra-ai/cuda-foundations/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/open-infra-ai/cuda-foundations/ci.yml?style=flat-square&logo=github&label=CI" alt="CI"></a>
  <a href="https://open-infra-ai.github.io/cuda-foundations/"><img src="https://img.shields.io/github/actions/workflow/status/open-infra-ai/cuda-foundations/pages.yml?branch=master&style=flat-square&logo=githubpages&logoColor=white&label=Pages" alt="Pages"></a>
  <a href="https://github.com/open-infra-ai/cuda-foundations/releases"><img src="https://img.shields.io/github/v/release/open-infra-ai/cuda-foundations?style=flat-square&logo=github" alt="Release"></a>
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/C++-17%2F20-00599C?style=flat-square&logo=cplusplus" alt="C++">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

<div class="showcase-hero">
  <p class="showcase-kicker">这个仓库面向希望真正理解 CUDA 算子工程路径的读者：从第一个可运行 SGEMM，到可复用 kernel、进阶优化实验，以及轻量级推理系统拼装。</p>
  <div class="showcase-cta">
    <a href="https://open-infra-ai.github.io/cuda-foundations/">访问 GitHub Pages</a>
    <a href="docs/zh/index.md">查看文档</a>
    <a href="01-sgemm-tutorial/README.md">从模块 01 开始</a>
  </div>
</div>

<div class="showcase-metrics">
  <div><strong>4</strong><span>核心模块</span></div>
  <div><strong>2</strong><span>构建体系</span></div>
  <div><strong>1</strong><span>根目录变更日志</span></div>
</div>

## 这个项目解决什么问题

很多 CUDA 教程要么过于碎片化，要么大到难以整体理解。CUDA Kernel Academy 的定位是：

- **模块 01**：直接围绕 SGEMM 学优化路径
- **模块 02**：把这些思路整理成可复用算子库形态
- **模块 03**：继续深入更复杂的 CUDA / HPC 主题
- **模块 04**：把 kernel、内存、stream、配置组合成小型推理系统

## 仓库状态

教学仓库，状态 **active**（正确性修复与教学打磨；不再新增模块）。四个模块已完成。2026-08-23 在 RTX 3060 Laptop（sm_86、CUDA 12.0）上运行 `ctest --preset default`，结果为 **261/261 通过**。不再新增模块；后续精力投入 [LEARNING_PATH.md](https://github.com/open-infra-ai/open-infra-ai/blob/master/LEARNING_PATH.md) 中记录的后续项目（如 `tiny-llm` 与 `paged-infer`）。

## 仓库收敛边界

CUDA Kernel Academy 是 `sgemm-optimization`、`modern-ai-kernels`、`hpc-ai-optimization-lab` 和 `mini-inference-engine` 四条重叠学习路径的唯一维护入口。仍有价值的实现和验证已经进入对应 Academy 模块，重复仓库直接退场，不再保留平行维护入口。

原 `llm-speed` 的 GEMM 学习路径由本仓库覆盖，仍维护的深度 FlashAttention 实现位于 [`cuflash-attn`](https://github.com/open-infra-ai/cuflash-attn)。本仓库不会搬入每一种历史实现，只为每个教学角色保留一条清晰路径。

## 项目边界（IN / OUT）

**IN（本仓库负责）**：

- CUDA 编程模型与 kernel 编写（线程/块/共享内存/同步）
- GEMM 优化阶梯（naive → tiled → bank-conflict-free → double buffering → WMMA）
- 算子库设计（header-only 布局、operator surface）
- profiling 与性能分析方法

**OUT（明确不做，见对应仓库）**：

- 生产级推理运行时（模型加载/采样/生成）→ [tiny-llm](https://github.com/open-infra-ai/tiny-llm)
- 完整 FlashAttention 前后向与优化 → [cuflash-attn](https://github.com/open-infra-ai/cuflash-attn)
- Serving / 分页 KV / continuous batching → [paged-infer](https://github.com/open-infra-ai/paged-infer)
- 本仓库 [04-inference-engine](04-inference-engine/README.md) 为**教学预览**，非独立作品
- 本仓库 [03-hpc-advanced](03-hpc-advanced/README.md) 的 FlashAttention 核是教学简化版，权威实现见 cuflash-attn

## 项目地图

| 模块 | 你会学到什么 | 构建方式 |
| --- | --- | --- |
| [01-sgemm-tutorial](01-sgemm-tutorial/README.md) | tiled SGEMM、bank conflict、double buffering、WMMA | 根 CMake + 可选 `Makefile` |
| [02-tensorcraft-core](02-tensorcraft-core/README.md) | 可复用 kernel API、header-only 组织方式、算子接口（历史模块名 TensorCraft，不是独立产品） | 根目录 / 模块 CMake |
| [03-hpc-advanced](03-hpc-advanced/README.md) | 进阶优化主题、实验性 CUDA 12+ 能力（内含教学用 FA 简化核） | 根目录 / 模块 CMake |
| [04-inference-engine](04-inference-engine/README.md)（教学预览） | tensor plumbing、memory pool、stream、轻量推理流程 | 根目录 / 模块 CMake |

## 建议从哪里开始

| 目标 | 入口 |
| --- | --- |
| 从第一性原理理解 CUDA 优化 | [01-sgemm-tutorial](01-sgemm-tutorial/README.md) |
| 看一个可复用算子库怎么组织 | [02-tensorcraft-core](02-tensorcraft-core/README.md) |
| 学更进阶的 CUDA / HPC 主题 | [03-hpc-advanced](03-hpc-advanced/README.md) |
| 看 kernel 如何落到小系统里 | [04-inference-engine](04-inference-engine/README.md) |
| 了解构建、验证、协作流程 | [docs/zh/index.md](docs/zh/index.md) |

如果目标是从初学者逐步转向 AI Infra 岗位，请按 [AI Infra 初学者学习路径](https://github.com/open-infra-ai/open-infra-ai/blob/master/LEARNING_PATH.md) 串联 CUDA、Triton、FlashAttention、推理运行时和 Serving 五个主仓。该文档也统一保留性能优化与不变量测试的方法。

## 快速开始

```bash
git clone https://github.com/open-infra-ai/cuda-foundations.git
cd cuda-foundations

cmake --list-presets
cmake --preset default
cmake --build --preset default
ctest --preset default
```

独立教程模块：

```bash
cd 01-sgemm-tutorial
make GPU_ARCH=sm_86
make test
```

## 关于构建与验证

- 根 CMake 图覆盖 `01-sgemm-tutorial`、`02-tensorcraft-core`、`03-hpc-advanced`、`04-inference-engine`、`common` 和 `examples`
- `01-sgemm-tutorial/Makefile` 仍可用于快速独立运行，但测试与 CI 使用根 CMake 图
- GitHub Actions 仅做 CPU-safe 检查
- 真正的 CUDA 编译与运行验证应在本地 GPU 机器上完成

## 文档入口

- [文档总览](docs/zh/index.md)
- [后续开发计划（供执行模型使用）](DEV_PLAN.md)
- [快速开始](docs/zh/guides/getting-started.md)
- [开发流程](docs/zh/guides/workflow.md)
- [编码风格](docs/zh/reference/coding-style.md)
- [故障排除](docs/zh/reference/troubleshooting.md)
- [变更日志](CHANGELOG.md)
- [English docs home](docs/en/index.md)
- [贡献指南](CONTRIBUTING.md)

## 环境要求

| 组件 | 最低版本 | 推荐版本 |
| --- | --- | --- |
| CUDA Toolkit | 12.0 | 12.x |
| CMake | 3.20 | 3.24+ |
| 编译器 | GCC 9 / Clang 10 | GCC 11+ |
| GPU | Volta (sm_70) | Ampere / Ada / Hopper |

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [Simon Boehm's GEMM Tutorial](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)

## 引用

```text
@misc{cuda-foundations,
  author = {CUDA Kernel Academy Contributors},
  title = {CUDA Kernel Academy},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/open-infra-ai/cuda-foundations}
}
```

## License

MIT
