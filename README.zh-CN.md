# CUDA Kernel Academy

<p align="center">
  <b>从零到极致：系统性学习 CUDA 高性能算子开发</b><br>
  <b>From Zero to Hero: Systematic CUDA High-Performance Kernel Development</b>
</p>

<p align="center">
  <a href="https://github.com/LessUp/cuda-kernel-academy/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/LessUp/cuda-kernel-academy/ci.yml?style=flat-square&logo=github&label=CI" alt="CI"></a>
  <a href="https://lessup.github.io/cuda-kernel-academy/"><img src="https://img.shields.io/github/actions/workflow/status/LessUp/cuda-kernel-academy/pages.yml?branch=master&style=flat-square&logo=githubpages&logoColor=white&label=Docs" alt="Docs"></a>
  <a href="https://github.com/LessUp/cuda-kernel-academy/releases"><img src="https://img.shields.io/github/v/release/LessUp/cuda-kernel-academy?style=flat-square&logo=github" alt="Release"></a>
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/C++-17%2F20-00599C?style=flat-square&logo=cplusplus" alt="C++">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License"></a>
  <a href="https://github.com/LessUp/cuda-kernel-academy/stargazers"><img src="https://img.shields.io/github/stars/LessUp/cuda-kernel-academy?style=flat-square&logo=github" alt="Stars"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

---

一个系统性的 CUDA 学习仓库，涵盖矩阵乘法、可复用算子库、高级优化技术和轻量级推理引擎。从 SGEMM 基础到 Tensor Core 优化，掌握 GPU 编程的完整路径。

## 项目特色

| 特性 | 说明 |
|------|------|
| **渐进式学习路径** | 4 个相互关联的模块，从入门到生产级 |
| **性能导向** | 与 cuBLAS 对比的真实性能基准，非玩具示例 |
| **现代 C++** | 利用 C++17/20 特性编写简洁、安全的 GPU 代码 |
| **生产级模式** | Header-only 库设计、内存池、流管理 |
| **多架构支持** | 支持 Volta (sm_70) 到 Hopper (sm_90) |

## 项目结构

| # | 模块 | 说明 | 构建方式 |
|---|------|------|----------|
| 01 | [SGEMM Tutorial](./01-sgemm-tutorial/) | 逐步优化 SGEMM 的入门教程 | 独立 Makefile |
| 02 | [TensorCraft Core](./02-tensorcraft-core/) | Header-only 高性能算子库 | CMake |
| 03 | [HPC Advanced](./03-hpc-advanced/) | 进阶 CUDA / HPC 优化示例 | CMake |
| 04 | [Inference Engine](./04-inference-engine/) | 轻量级推理引擎框架 | CMake |

## 推荐学习路径

```text
01-sgemm-tutorial (1-2 周)
        ↓  掌握共享内存、bank conflict、WMMA
02-tensorcraft-core (2-3 周)
        ↓  构建可复用算子、API 设计
03-hpc-advanced (3-4 周)
        ↓  CUDA 13 特性、FlashAttention
04-inference-engine (2-3 周)
        ↓  完整推理框架
```

**前置要求**：C/C++ 基础、线性代数基础。有 CUDA 经验更好，但非必需。

## 快速开始

```bash
git clone https://github.com/LessUp/cuda-kernel-academy.git
cd cuda-kernel-academy

cmake --preset default
cmake --build --preset default
ctest --preset default
```

查看可用预设：

```bash
cmake --list-presets
```

## 构建说明

- 根目录 CMake 负责构建 `02-tensorcraft-core`、`03-hpc-advanced`、`04-inference-engine`、`common` 与 `examples`。
- `01-sgemm-tutorial` 是独立教程模块，不参与根目录 CMake 构建，使用自身的 `Makefile`。
- GitHub Actions 目前只运行 CPU-safe 检查（格式、文档、链接、preset 合法性）；完整 CUDA 构建/测试请在本地 GPU 环境执行。

## 常用构建选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `BUILD_TENSORCRAFT` | ON | 构建 02-tensorcraft-core |
| `BUILD_HPC_ADVANCED` | ON | 构建 03-hpc-advanced |
| `BUILD_INFERENCE_ENGINE` | ON | 构建 04-inference-engine |
| `BUILD_EXAMPLES` | ON | 构建示例 |
| `BUILD_TESTS` | ON | 构建测试 |
| `BUILD_BENCHMARKS` | ON | 构建 benchmark |
| `BUILD_PYTHON_BINDINGS` | OFF | 构建可选 Python 绑定 |

## 环境要求

| 组件 | 最低版本 | 推荐版本 |
|------|----------|----------|
| CUDA Toolkit | 11.0 | 12.x |
| CMake | 3.20 | 3.24+ |
| 编译器 | GCC 9 / Clang 10 | GCC 11+ |
| GPU | Volta (sm_70) | Ampere/Ada (sm_80+) |

**支持的架构**：

| 架构 | sm | GPU 示例 |
|------|-----|----------|
| Volta | 70 | V100 |
| Turing | 75 | RTX 2080, T4 |
| Ampere | 80, 86 | A100, RTX 3090 |
| Ada | 89 | RTX 4090, L40 |
| Hopper | 90 | H100 |

## 文档

- [安装指南](docs/INSTALLATION.md)
- [代码风格](docs/CODING_STYLE.md)
- [集成示例](docs/integration_examples.md)
- [常见问题](docs/FAQ.md)
- [故障排除](docs/TROUBLESHOOTING.md)
- [贡献指南](CONTRIBUTING.md)

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass) - CUDA 线性代数模板库
- [Simon Boehm's GEMM Tutorial](https://siboehm.com/articles/22/CUDA-MMM) - 优秀的优化教程
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/) - 最新技术和最佳实践

## 引用

如果在研究或工作中使用了本项目：

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
