# 安装指南

本文档给出当前仓库真实可用的安装与构建方式。

## 系统要求

| 项目 | 最低版本 | 推荐版本 |
|------|----------|----------|
| CUDA Toolkit | 12.0 | 12.x 最新稳定版 |
| CMake | 3.20 | 3.24+ |
| GCC / Clang | GCC 9 / Clang 10 | GCC 11+ |
| Python | 3.8 | 3.10+ |
| GPU | Volta (sm_70) | Ampere / Ada / Hopper |

## 克隆仓库

```bash
git clone https://github.com/LessUp/cuda-kernel-academy.git
cd cuda-kernel-academy
```

## 推荐方式：使用 CMake Presets

根目录的推荐构建入口是 `CMakePresets.json`。

```bash
# 查看可用预设
cmake --list-presets

# 默认构建
cmake --preset default
cmake --build --preset default
ctest --preset default
```

常用预设：

- `default`: 默认 Release 构建
- `debug`: Debug 构建
- `minimal`: 仅构建 TensorCraft Core 的精简构建
- `full`: 构建更多可选内容
- `ampere` / `hopper`: 针对特定架构设置 CUDA arch

## 手动 CMake 构建

如果不使用 presets，也可以手动配置：

```bash
cmake -S . -B build/manual -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="80;86"

cmake --build build/manual -j$(nproc)
ctest --test-dir build/manual --output-on-failure
```

## 根目录可用构建选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `BUILD_TENSORCRAFT` | ON | 构建 `02-tensorcraft-core` |
| `BUILD_HPC_ADVANCED` | ON | 构建 `03-hpc-advanced` |
| `BUILD_INFERENCE_ENGINE` | ON | 构建 `04-inference-engine` |
| `BUILD_EXAMPLES` | ON | 构建 `examples` |
| `BUILD_TESTS` | ON | 构建测试 |
| `BUILD_BENCHMARKS` | ON | 构建 benchmark |
| `BUILD_PYTHON_BINDINGS` | OFF | 构建可选 Python 绑定 |

## 单独构建各模块

### 01-sgemm-tutorial

`01-sgemm-tutorial` 是独立教程模块，**不参与根目录 CMake 构建**。

```bash
cd 01-sgemm-tutorial
make GPU_ARCH=sm_86
./build/sgemm_benchmark

# 运行测试（需要系统可用的 GoogleTest）
make test
```

### 02-tensorcraft-core

```bash
cd 02-tensorcraft-core
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

### 03-hpc-advanced

```bash
cd 03-hpc-advanced
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

### 04-inference-engine

```bash
cd 04-inference-engine
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

## 文档构建

```bash
npm ci
npx honkit build . _site
```

## 本地质量检查

```bash
pre-commit run --all-files
cmake --list-presets
```

## CI 与 GPU 验证说明

GitHub-hosted CI 当前主要执行 CPU-safe 检查：

- pre-commit
- 文档构建
- 链接检查
- preset 合法性检查

完整 CUDA 编译和运行测试请在本地或自有 GPU 环境完成。
