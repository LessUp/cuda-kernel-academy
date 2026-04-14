# CUDA Kernel Academy

[English](README.md) | 简体中文

<p align="center">
  <b>从基础 SGEMM 到轻量推理引擎的系统化 CUDA 学习仓库</b>
</p>

<p align="center">
  <a href="https://github.com/LessUp/cuda-kernel-academy/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/LessUp/cuda-kernel-academy/ci.yml?style=flat-square&logo=github&label=CI" alt="CI"></a>
  <a href="https://lessup.github.io/cuda-kernel-academy/"><img src="https://img.shields.io/github/actions/workflow/status/LessUp/cuda-kernel-academy/pages.yml?branch=master&style=flat-square&logo=githubpages&logoColor=white&label=Docs" alt="Docs"></a>
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/C++-17%2F20-00599C?style=flat-square&logo=cplusplus" alt="C++">
</p>

## 项目结构

| # | 模块 | 说明 | 构建方式 |
|---|------|------|----------|
| 01 | [SGEMM Tutorial](./01-sgemm-tutorial/) | 逐步优化 SGEMM 的入门教程 | 独立 Makefile |
| 02 | [TensorCraft Core](./02-tensorcraft-core/) | Header-only 高性能算子库 | CMake |
| 03 | [HPC Advanced](./03-hpc-advanced/) | 进阶 CUDA / HPC 优化示例 | CMake |
| 04 | [Inference Engine](./04-inference-engine/) | 轻量级推理引擎框架 | CMake |

## 推荐学习路径

```text
01-sgemm-tutorial
    ↓
02-tensorcraft-core
    ↓
03-hpc-advanced
    ↓
04-inference-engine
```

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

- CUDA Toolkit 12.x+
- CMake 3.20+
- GCC / Clang（支持 C++17/20）
- SM 7.0+ NVIDIA GPU

## 文档

- [安装指南](docs/INSTALLATION.md)
- [代码风格](docs/CODING_STYLE.md)
- [集成示例](docs/integration_examples.md)
- [常见问题](docs/FAQ.md)
- [故障排除](docs/TROUBLESHOOTING.md)
- [贡献指南](CONTRIBUTING.md)

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [Simon Boehm's CUDA Tutorials](https://siboehm.com/articles/22/CUDA-MMM)

## License

MIT License
