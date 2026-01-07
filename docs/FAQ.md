# ❓ 常见问题解答 | FAQ

本文档回答关于 CUDA Kernel Academy 项目的常见问题。

---

## 项目概述

### Q: 这个项目是什么？

**A:** CUDA Kernel Academy 是一个系统性的 CUDA 高性能计算学习项目，包含从入门到进阶的完整学习路径。项目整合了 4 个相互关联的子模块：

1. **01-sgemm-tutorial** - SGEMM 优化入门教程
2. **02-tensorcraft-core** - 高性能算子核心库
3. **03-hpc-advanced** - CUDA 13 新特性和进阶优化
4. **04-inference-engine** - 推理引擎框架

### Q: 这个项目适合谁？

**A:** 适合以下人群：

- 🎓 **CUDA 初学者**：想系统学习 GPU 编程
- 💼 **深度学习工程师**：想了解底层算子实现
- 🔬 **HPC 研究人员**：想学习最新的 CUDA 优化技术
- 🛠️ **框架开发者**：想构建自己的推理引擎

### Q: 学习这个项目需要什么前置知识？

**A:** 建议具备：

- C/C++ 编程基础
- 基本的线性代数知识（矩阵乘法）
- 了解计算机体系结构概念（缓存、内存层次）
- 有 CUDA 基础更好，但不是必须

---

## 学习路径

### Q: 应该从哪个模块开始学习？

**A:** 推荐按以下顺序：

```
01-sgemm-tutorial (1-2周)
        ↓
02-tensorcraft-core (2-3周)
        ↓
03-hpc-advanced 或 04-inference-engine (3-4周)
```

如果你是完全的 CUDA 新手，建议先完成 NVIDIA 官方的 [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 入门章节。

### Q: 每个模块需要多长时间？

**A:** 取决于你的背景：

| 模块 | 初学者 | 有经验者 |
|------|--------|----------|
| 01-sgemm-tutorial | 2-3 周 | 1 周 |
| 02-tensorcraft-core | 3-4 周 | 2 周 |
| 03-hpc-advanced | 4-6 周 | 2-3 周 |
| 04-inference-engine | 4-6 周 | 2-3 周 |

### Q: 可以跳过某些模块吗？

**A:** 可以，但建议：

- **01-sgemm-tutorial** 是基础，强烈建议完成
- **02-tensorcraft-core** 是核心库，04 依赖它
- **03-hpc-advanced** 和 **04-inference-engine** 可以并行学习

---

## 技术问题

### Q: 支持哪些 GPU？

**A:** 支持 NVIDIA Volta 及更新架构：

| 架构 | 计算能力 | 示例 GPU |
|------|----------|----------|
| Volta | sm_70 | V100 |
| Turing | sm_75 | RTX 2080, T4 |
| Ampere | sm_80, sm_86 | A100, RTX 3090 |
| Ada | sm_89 | RTX 4090, L40 |
| Hopper | sm_90 | H100 |

**注意**：03-hpc-advanced 的部分功能（TMA、Cluster）需要 Hopper (sm_90) 架构。

### Q: 支持哪些 CUDA 版本？

**A:**

| 模块 | 最低 CUDA 版本 | 推荐版本 |
|------|----------------|----------|
| 01-sgemm-tutorial | 11.0 | 12.0+ |
| 02-tensorcraft-core | 11.0 | 12.0+ |
| 03-hpc-advanced | 12.0 | 12.3+ |
| 04-inference-engine | 11.0 | 12.0+ |

### Q: 可以在 Windows 上运行吗？

**A:** 理论上可以，但我们主要在 Linux 上测试。Windows 用户建议：

1. 使用 WSL2 + CUDA
2. 使用 Docker Desktop with GPU support
3. 使用 Visual Studio 2022 + CUDA Toolkit

### Q: 可以在 Mac 上运行吗？

**A:** 不支持。Apple Silicon Mac 没有 NVIDIA GPU，Intel Mac 的 NVIDIA GPU 支持已停止。

### Q: 为什么我的 kernel 比 cuBLAS 慢？

**A:** 这是正常的！cuBLAS 经过多年优化，包含：

- 针对每种 GPU 架构的手工调优
- 自动选择最优算法
- 汇编级优化

本项目的目标是**学习原理**，而非替代 cuBLAS。通过学习，你可以：

- 理解 cuBLAS 为什么快
- 为特定场景编写定制 kernel
- 优化 cuBLAS 不支持的操作

### Q: Tensor Core 和普通 CUDA Core 有什么区别？

**A:**

| 特性 | CUDA Core | Tensor Core |
|------|-----------|-------------|
| 操作 | 标量 FMA | 矩阵 FMA (4x4 或更大) |
| 精度 | FP32, FP64 | FP16, BF16, TF32, FP8, INT8 |
| 吞吐量 | 1 FMA/cycle/core | 64+ FMA/cycle/core |
| 编程 | 直接使用 | WMMA/MMA API |

Tensor Core 在深度学习推理中可提供 3-10x 加速。

---

## 构建和安装

### Q: CMake 配置失败怎么办？

**A:** 常见解决方案：

```bash
# 1. 确保 CUDA 在 PATH 中
export PATH=/usr/local/cuda/bin:$PATH

# 2. 指定 CUDA 编译器
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# 3. 指定 GPU 架构
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86

# 4. 清理重建
rm -rf build && mkdir build && cd build && cmake ..
```

详见 [故障排除指南](TROUBLESHOOTING.md)。

### Q: 如何只构建某个模块？

**A:**

```bash
# 方法 1: 使用 CMake 选项
cmake .. -DBUILD_SGEMM_TUTORIAL=ON \
         -DBUILD_TENSORCRAFT=OFF \
         -DBUILD_HPC_ADVANCED=OFF \
         -DBUILD_INFERENCE_ENGINE=OFF

# 方法 2: 直接进入模块目录构建
cd 02-tensorcraft-core
mkdir build && cd build
cmake .. && make -j
```

### Q: 如何加速编译？

**A:**

```bash
# 1. 使用 Ninja 替代 Make
cmake .. -G Ninja
ninja -j$(nproc)

# 2. 使用 ccache
sudo apt install ccache
export CUDA_NVCC_EXECUTABLE="ccache nvcc"

# 3. 减少 GPU 架构数量
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86  # 只编译你的 GPU

# 4. 并行编译
make -j$(nproc)
```

---

## 贡献相关

### Q: 如何贡献代码？

**A:**

1. Fork 仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m "feat: add your feature"`
4. 推送分支：`git push origin feature/your-feature`
5. 创建 Pull Request

详见 [CONTRIBUTING.md](../CONTRIBUTING.md)。

### Q: 代码风格有什么要求？

**A:**

- 使用项目提供的 `.clang-format` 格式化代码
- 遵循 [代码风格指南](CODING_STYLE.md)
- 使用 `pre-commit` 进行本地检查

```bash
# 安装 pre-commit
pip install pre-commit
pre-commit install

# 手动运行检查
pre-commit run --all-files
```

### Q: 可以添加新的算子吗？

**A:** 当然！欢迎贡献新算子。建议：

1. 先创建 Issue 讨论设计
2. 参考现有算子的实现风格
3. 包含完整的测试和文档
4. 添加 benchmark 对比性能

### Q: 发现 Bug 怎么报告？

**A:** 创建 GitHub Issue，包含：

- 问题描述
- 复现步骤
- 环境信息（OS、CUDA 版本、GPU 型号）
- 错误日志

使用我们的 [Bug Report 模板](https://github.com/yourusername/cuda-kernel-academy/issues/new?template=bug_report.yml)。

---

## 许可和使用

### Q: 这个项目的许可证是什么？

**A:** MIT License。你可以：

- ✅ 商业使用
- ✅ 修改
- ✅ 分发
- ✅ 私人使用

唯一要求是保留版权声明。

### Q: 可以在商业项目中使用吗？

**A:** 可以。MIT 许可证允许商业使用。但请注意：

- 本项目主要用于**学习目的**
- 生产环境建议使用 cuBLAS、cuDNN 等成熟库
- 如果使用本项目代码，请保留版权声明

### Q: 可以用于教学吗？

**A:** 非常欢迎！如果你在课程中使用本项目：

- 欢迎 Star 和 Fork
- 可以在 Issue 中分享你的教学经验
- 欢迎贡献教学相关的改进

---

## 其他问题

### Q: 有官方交流群吗？

**A:** 目前没有官方群组。你可以：

- 在 GitHub Issues 中提问
- 在 GitHub Discussions 中讨论
- 关注项目更新

### Q: 项目会持续更新吗？

**A:** 是的！计划中的更新包括：

- 支持更多 GPU 架构
- 添加更多算子实现
- 改进文档和教程
- 添加更多示例

### Q: 如何引用这个项目？

**A:** 如果在学术论文中使用，可以引用：

```bibtex
@misc{cuda-kernel-academy,
  author = {CUDA Kernel Academy Contributors},
  title = {CUDA Kernel Academy: A Comprehensive Learning Path for High-Performance CUDA Kernel Development},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/cuda-kernel-academy}
}
```

---

## 还有问题？

如果你的问题没有在这里找到答案：

1. 查看 [故障排除指南](TROUBLESHOOTING.md)
2. 搜索 [GitHub Issues](https://github.com/yourusername/cuda-kernel-academy/issues)
3. 创建新的 Issue 或 Discussion

我们很乐意帮助！🚀
