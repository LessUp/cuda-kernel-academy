# 故障排除指南

本文档收集 CUDA Kernel Academy 项目的常见问题及解决方案。

## 编译错误

### nvcc: command not found

```bash
# 添加 CUDA 到 PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### GCC 版本不兼容

| CUDA 版本 | 支持的 GCC |
|-----------|------------|
| CUDA 11.x | GCC ≤ 10 |
| CUDA 12.0-12.2 | GCC ≤ 12 |
| CUDA 12.3+ | GCC ≤ 13 |

```bash
cmake .. -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11
```

### GPU 架构不匹配

```bash
# 查询 GPU 计算能力
nvidia-smi --query-gpu=compute_cap --format=csv

# 重新编译
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86  # RTX 3090
```

---

## 运行时错误

### CUDA error: out of memory

```bash
nvidia-smi  # 检查显存使用
```

### CUDA error: illegal memory access

```bash
compute-sanitizer --tool memcheck ./your_program
```

常见原因：数组越界、使用已释放内存、线程索引错误。

### 设备端 assert 失败

```bash
export CUDA_LAUNCH_BLOCKING=1
./your_program
```

---

## CMake 配置问题

### Could not find CUDA

```bash
export CUDA_HOME=/usr/local/cuda
cmake .. -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
```

### CMake 版本过低

```bash
sudo snap install cmake --classic  # Ubuntu
```

---

## 模块特定问题

### 03-hpc-advanced 编译失败

原因：需要 CUDA 12.0+ 和 C++20 支持。

```bash
nvcc --version  # 确认 CUDA 版本
```

### 04-inference-engine 链接错误

原因：依赖 tensorcraft-core。

```bash
cmake .. -DBUILD_TENSORCRAFT=ON
```

---

## 调试工具

| 工具 | 用途 | 命令 |
|------|------|------|
| compute-sanitizer | 内存检测 | `compute-sanitizer ./program` |
| nsys | 系统分析 | `nsys profile ./program` |
| ncu | Kernel 分析 | `ncu ./program` |

---

## 获取帮助

1. 搜索 [GitHub Issues](https://github.com/LessUp/cuda-kernel-academy/issues)
2. 查看 [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
3. 创建新 Issue，提供：错误信息、OS、CUDA 版本、GPU 型号

---

## 更多资源

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-best-practices-guide/)
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)
