# 📚 CUDA Kernel Academy 示例代码

本目录包含 CUDA Kernel Academy 的独立示例代码，帮助你快速上手各个模块。

## 示例列表

| 示例 | 说明 | 难度 |
|------|------|------|
| [01_basic_gemm](01_basic_gemm/) | 基础 GEMM 矩阵乘法 | ⭐ |
| [02_tensor_operations](02_tensor_operations/) | Tensor 基本操作 | ⭐⭐ |

## 快速开始

### 构建所有示例

```bash
# 从项目根目录
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=ON
cmake --build .

# 或使用 CMake Presets
cmake --preset default
cmake --build build
```

### 构建单个示例

```bash
cd examples/01_basic_gemm
mkdir build && cd build
cmake ..
make
./basic_gemm
```

## 示例说明

### 01_basic_gemm

演示基础的 GEMM（通用矩阵乘法）实现：

- Naive 三层循环实现
- Tiled 共享内存优化
- 性能对比和验证

**学习要点**：
- CUDA kernel 基本结构
- 共享内存使用
- 内存合并访问

### 02_tensor_operations

演示 Tensor 类的基本操作：

- Tensor 创建和初始化
- 元素级操作（加、减、乘）
- 内存管理（Host/Device 传输）

**学习要点**：
- RAII 内存管理
- CUDA 流使用
- 错误处理

## 运行要求

- CUDA Toolkit 11.0+
- CMake 3.20+
- 支持 sm_70+ 的 NVIDIA GPU

## 示例代码风格

所有示例遵循以下原则：

1. **自包含**：每个示例可独立编译运行
2. **注释详尽**：关键代码都有中英文注释
3. **渐进式**：从简单到复杂，循序渐进
4. **可验证**：包含正确性验证代码

## 添加新示例

欢迎贡献新示例！请遵循：

1. 创建新目录 `examples/XX_example_name/`
2. 包含 `CMakeLists.txt` 和源代码
3. 添加详细注释
4. 更新本 README

## 相关资源

- [安装指南](../docs/INSTALLATION.md)
- [代码风格](../docs/CODING_STYLE.md)
- [故障排除](../docs/TROUBLESHOOTING.md)
