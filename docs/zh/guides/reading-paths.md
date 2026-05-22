---
outline: [2, 3]
---

# 阅读路径

如果你不想按目录顺序阅读仓库，可以从这里选一条路径开始。

## 路径 1：第一次接触仓库

适合想先建立全局认知、再决定看哪些代码的读者。

1. [快速开始](./getting-started.md)
2. [01-SGEMM 教程](/zh/modules/01-sgemm)
3. [系统架构白皮书](/zh/whitepaper/architecture)
4. [路线图](/zh/roadmap)

## 路径 2：优先理解 kernel 优化

适合先看性能机制、再看系统组织的读者。

1. [01-SGEMM 教程](/zh/modules/01-sgemm)
2. [SGEMM 优化白皮书](/zh/whitepaper/sgemm-optimization)
3. [03-HPC 进阶](/zh/modules/03-hpc)
4. [Benchmarks](/zh/benchmarks/)

## 路径 3：优先理解可复用 CUDA 库设计

适合关注接口、库形态和复用方式的读者。

1. [02-TensorCraft Core](/zh/modules/02-tensorcraft)
2. [TensorCraft 设计](/zh/whitepaper/tensorcraft-design)
3. [模块架构](/zh/modules/02/architecture)
4. [API 参考](/zh/modules/02/api-reference)

## 路径 4：优先理解推理系统集成

适合想看 kernel 如何落到端到端执行链路里的读者。

1. [04-Inference Engine](/zh/modules/04-inference)
2. [推理引擎设计](/zh/whitepaper/inference-engine-design)
3. [快速开始](/zh/modules/04/quick-start)
4. [性能调优](/zh/modules/04/performance-tuning)

## 验证现实

- 文档和 Pages 构建属于 CPU-safe 检查。
- CUDA 编译与运行验证仍应在真实 GPU 机器上完成。
- 仓库变更流程和验证命令见[开发流程](./workflow.md)。
