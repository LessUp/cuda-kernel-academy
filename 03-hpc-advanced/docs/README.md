# 📚 HPC-AI-Optimization-Lab 文档

欢迎阅读 HPC-AI-Optimization-Lab 的技术文档！

## 文档目录

### 核心优化技术

| 文档 | 描述 | 难度 |
|------|------|------|
| [01_gemm_optimization.md](01_gemm_optimization.md) | GEMM 7步优化详解 | ⭐⭐⭐⭐ |
| [02_memory_optimization.md](02_memory_optimization.md) | 访存优化技术 | ⭐⭐ |
| [03_reduction_optimization.md](03_reduction_optimization.md) | 归约优化技术 | ⭐⭐⭐ |
| [04_flash_attention.md](04_flash_attention.md) | FlashAttention 原理与实现 | ⭐⭐⭐⭐ |
| [05_cuda13_features.md](05_cuda13_features.md) | CUDA 13 & Hopper 新特性 | ⭐⭐⭐⭐⭐ |

## 学习路线

### 初学者 (1-2 周)

```
02_memory_optimization.md → 03_reduction_optimization.md → 01_gemm_optimization.md (Step 1-4)
```

1. 先学习访存优化基础
2. 掌握归约操作
3. 理解 GEMM 的基础优化

### 进阶 (2-4 周)

```
01_gemm_optimization.md (Step 5-7) → 04_flash_attention.md
```

1. 学习 Tensor Core 使用
2. 理解 FlashAttention 原理

### 专家 (持续学习)

```
05_cuda13_features.md → 阅读 CUTLASS 源码 → 研究最新论文
```

1. 掌握 Hopper 架构特性
2. 深入研究 CUTLASS 实现
3. 跟踪学术前沿

## 快速参考

### 常用优化技术

| 技术 | 适用场景 | 预期提升 |
|------|----------|----------|
| 合并访问 | 所有 Kernel | 2-10× |
| 向量化 (float4) | Elementwise | 1.5-2× |
| Shared Memory | 数据复用 | 2-5× |
| Warp Shuffle | 归约操作 | 5-10× |
| Tensor Core | 矩阵乘法 | 10-20× |
| TMA | 大块数据搬运 | 1.5-2× |

### 性能分析工具

```bash
# Nsight Compute - Kernel 分析
ncu --set full -o profile ./your_app

# Nsight Systems - 系统级分析
nsys profile -o timeline ./your_app

# 查看 Roofline
ncu --set roofline -o roofline ./your_app
```

### 常见问题

**Q: 为什么我的 Kernel 性能不好？**

1. 检查内存访问模式 (是否合并)
2. 检查 Occupancy (是否太低)
3. 检查 Bank Conflict (Shared Memory)
4. 使用 Nsight Compute 分析瓶颈

**Q: 什么时候使用 Tensor Core？**

- 矩阵乘法 (GEMM)
- 卷积 (通过 Implicit GEMM)
- Attention 计算
- 数据类型: FP16, BF16, FP8, INT8

**Q: 如何选择 Block 大小？**

- 通常 128-256 线程
- 考虑 Shared Memory 使用
- 考虑寄存器压力
- 使用 Occupancy Calculator

## 贡献文档

欢迎贡献更多文档！请遵循以下格式：

1. 使用 Markdown 格式
2. 包含代码示例
3. 添加性能对比数据
4. 引用相关论文/资料

## 参考资源

### 官方文档

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass/tree/main/media/docs)

### 论文

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Online Softmax](https://arxiv.org/abs/2112.05682)

### 博客

- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)

---

Happy Learning! 🚀
