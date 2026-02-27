# 📚 CUDA Kernel Academy 文档

欢迎来到 CUDA Kernel Academy 文档中心！

## 文档索引

### 入门指南

- [主 README](../README.md) - 项目概述和快速开始
- [安装指南](INSTALLATION.md) - 详细的安装和构建说明
- [常见问题](FAQ.md) - 常见问题解答
- [故障排除](TROUBLESHOOTING.md) - 常见问题及解决方案

### 项目规范

- [代码风格指南](CODING_STYLE.md) - 统一的代码风格和技术标准
- [Integration Examples](integration_examples.md) - 如何在你的项目中集成各模块

### 示例代码

- [示例目录](../examples/README.md) - 独立可运行的示例代码

### 各模块文档

| 模块 | 文档位置 | 说明 |
|------|----------|------|
| 01-sgemm-tutorial | [README](../01-sgemm-tutorial/README.md) | SGEMM 优化教程 |
| 02-tensorcraft-core | [docs/](../02-tensorcraft-core/docs/) | 核心算子库文档 |
| 03-hpc-advanced | [docs/](../03-hpc-advanced/docs/) | 进阶优化文档 |
| 04-inference-engine | [docs/](../04-inference-engine/docs/) | 推理引擎文档 |

### 技术深度文档

#### 02-tensorcraft-core

- [architecture.md](../02-tensorcraft-core/docs/architecture.md) - 架构设计
- [api_reference.md](../02-tensorcraft-core/docs/api_reference.md) - API 参考
- [optimization_guide.md](../02-tensorcraft-core/docs/optimization_guide.md) - 优化指南
- [modern_cpp_cuda.md](../02-tensorcraft-core/docs/modern_cpp_cuda.md) - Modern C++ CUDA

#### 03-hpc-advanced

- [01_gemm_optimization.md](../03-hpc-advanced/docs/01_gemm_optimization.md) - GEMM 优化详解
- [02_memory_optimization.md](../03-hpc-advanced/docs/02_memory_optimization.md) - 内存优化
- [03_reduction_optimization.md](../03-hpc-advanced/docs/03_reduction_optimization.md) - Reduction 优化
- [04_flash_attention.md](../03-hpc-advanced/docs/04_flash_attention.md) - Flash Attention
- [05_cuda13_features.md](../03-hpc-advanced/docs/05_cuda13_features.md) - CUDA 13 新特性

#### 04-inference-engine

- [ARCHITECTURE.md](../04-inference-engine/docs/ARCHITECTURE.md) - 架构设计
- [API_REFERENCE.md](../04-inference-engine/docs/API_REFERENCE.md) - API 参考
- [QUICK_START.md](../04-inference-engine/docs/QUICK_START.md) - 快速开始
- [PERFORMANCE_TUNING.md](../04-inference-engine/docs/PERFORMANCE_TUNING.md) - 性能调优

## 学习路径建议

### 🎓 初学者 (1-2周)

1. 阅读主 README 了解项目结构
2. 进入 `01-sgemm-tutorial`，按顺序学习 5 种 GEMM 优化
3. 运行 benchmark，理解性能差异

### 📈 进阶者 (2-4周)

1. 学习 `02-tensorcraft-core` 的 API 设计
2. 阅读 `architecture.md` 理解设计思想
3. 尝试在自己的项目中集成 tensorcraft

### 🚀 专家 (4-8周)

1. 深入 `03-hpc-advanced` 的 CUDA 13 特性
2. 学习 Flash Attention 实现
3. 构建完整的推理引擎

## 贡献文档

欢迎贡献文档改进！请参考根目录的 [CONTRIBUTING.md](../CONTRIBUTING.md)。
