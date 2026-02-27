# 📦 安装指南 | Installation Guide

本文档提供 CUDA Kernel Academy 的详细安装说明。

## 系统要求

### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | NVIDIA Volta (sm_70) | NVIDIA Ampere (sm_80+) |
| GPU 显存 | 4 GB | 8 GB+ |
| 系统内存 | 8 GB | 16 GB+ |
| 磁盘空间 | 5 GB | 10 GB+ |

### 软件要求

| 软件 | 最低版本 | 推荐版本 | 说明 |
|------|----------|----------|------|
| CUDA Toolkit | 11.0 | 12.0+ | 核心依赖 |
| CMake | 3.20 | 3.24+ | 构建系统 |
| GCC | 9.0 | 11.0+ | C++ 编译器 |
| Clang | 10.0 | 15.0+ | 可选编译器 |
| Ninja | 1.10 | 1.11+ | 推荐构建工具 |
| Python | 3.8 | 3.10+ | Python 绑定 (可选) |

### 支持的 GPU 架构

| 架构 | 计算能力 | GPU 型号示例 |
|------|----------|--------------|
| Volta | sm_70 | V100 |
| Turing | sm_75 | RTX 2080, T4 |
| Ampere | sm_80, sm_86 | A100, RTX 3090 |
| Ada Lovelace | sm_89 | RTX 4090, L40 |
| Hopper | sm_90 | H100 |

---

## Linux 安装

### 1. 安装 CUDA Toolkit

#### Ubuntu/Debian

```bash
# 添加 NVIDIA 仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# 安装 CUDA Toolkit
sudo apt-get install cuda-toolkit-12-0

# 设置环境变量
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
nvcc --version
nvidia-smi
```

#### RHEL/CentOS/Rocky Linux

```bash
# 添加 NVIDIA 仓库
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# 安装 CUDA Toolkit
sudo dnf install cuda-toolkit-12-0

# 设置环境变量
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 2. 安装构建工具

```bash
# Ubuntu/Debian
sudo apt-get install cmake ninja-build git

# RHEL/CentOS
sudo dnf install cmake ninja-build git

# 验证版本
cmake --version
ninja --version
```

### 3. 克隆仓库

```bash
git clone https://github.com/LessUp/cuda-kernel-academy.git
cd cuda-kernel-academy
```

### 4. 构建项目

#### 使用 CMake Presets (推荐)

```bash
# 查看可用预设
cmake --list-presets

# 使用默认预设构建
cmake --preset default
cmake --build build

# 使用 Debug 预设
cmake --preset debug
cmake --build build
```

#### 手动 CMake 配置

```bash
mkdir build && cd build

# 基本配置
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80;86"

# 构建
cmake --build . -j$(nproc)

# 运行测试
ctest --output-on-failure
```

#### 构建选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `BUILD_SGEMM_TUTORIAL` | ON | 构建 01-sgemm-tutorial |
| `BUILD_TENSORCRAFT` | ON | 构建 02-tensorcraft-core |
| `BUILD_HPC_ADVANCED` | ON | 构建 03-hpc-advanced |
| `BUILD_INFERENCE_ENGINE` | ON | 构建 04-inference-engine |
| `BUILD_EXAMPLES` | ON | 构建示例代码 |
| `BUILD_TESTS` | ON | 构建测试 |
| `BUILD_BENCHMARKS` | ON | 构建性能测试 |

```bash
# 只构建特定模块
cmake .. -DBUILD_SGEMM_TUTORIAL=ON \
         -DBUILD_TENSORCRAFT=OFF \
         -DBUILD_HPC_ADVANCED=OFF \
         -DBUILD_INFERENCE_ENGINE=OFF
```

---

## Docker 安装

### 使用预构建镜像

```bash
# 拉取镜像
docker pull nvidia/cuda:12.0-devel-ubuntu22.04

# 运行容器
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    nvidia/cuda:12.0-devel-ubuntu22.04 \
    /bin/bash
```

### 使用项目 Dockerfile

```bash
# 构建镜像
cd 03-hpc-advanced/docker
docker build -t cuda-kernel-academy .

# 运行容器
docker run --gpus all -it --rm \
    -v $(pwd)/../..:/workspace \
    cuda-kernel-academy
```

### Docker Compose

```bash
cd 03-hpc-advanced/docker
docker-compose up -d
docker-compose exec cuda-dev bash
```

---

## 单独构建各模块

### 01-sgemm-tutorial

```bash
cd 01-sgemm-tutorial

# 使用 Makefile
make GPU_ARCH=sm_86
./build/sgemm_benchmark

# 或使用 CMake
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 02-tensorcraft-core

```bash
cd 02-tensorcraft-core
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure
```

### 03-hpc-advanced

```bash
cd 03-hpc-advanced
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
ctest --output-on-failure
```

### 04-inference-engine

```bash
cd 04-inference-engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./benchmark
```

---

## 验证安装

### 运行测试

```bash
# 从根目录
cd build
ctest --output-on-failure

# 运行特定测试
ctest -R gemm --output-on-failure
```

### 运行 Benchmark

```bash
# SGEMM benchmark
./01-sgemm-tutorial/build/sgemm_benchmark

# TensorCraft benchmark
./02-tensorcraft-core/build/benchmarks/gemm_benchmark

# Inference engine benchmark
./04-inference-engine/build/benchmark
```

### 检查 GPU 信息

```bash
# 使用 nvidia-smi
nvidia-smi

# 使用 CUDA 示例
./build/examples/01_basic_gemm/basic_gemm
```

---

## 常见安装问题

### CUDA 未找到

```bash
# 检查 CUDA 路径
which nvcc
echo $CUDA_HOME

# 设置 CUDA_HOME
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### CMake 找不到 CUDA

```bash
# 指定 CUDA 路径
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### GPU 架构不匹配

```bash
# 查询 GPU 计算能力
nvidia-smi --query-gpu=compute_cap --format=csv

# 指定正确的架构
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86
```

### 编译器版本不兼容

```bash
# 检查 GCC 版本
gcc --version

# 使用特定版本的 GCC
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++-11
```

---

## 下一步

安装完成后，建议：

1. 阅读 [快速开始](../README.md#-快速开始)
2. 运行 [示例代码](../examples/README.md)
3. 查看 [故障排除](TROUBLESHOOTING.md) 解决常见问题
4. 阅读 [FAQ](FAQ.md) 了解更多信息

---

## 获取帮助

如果遇到安装问题：

1. 查看 [故障排除指南](TROUBLESHOOTING.md)
2. 搜索 [GitHub Issues](https://github.com/LessUp/cuda-kernel-academy/issues)
3. 创建新的 Issue 并提供：
   - 操作系统版本
   - CUDA 版本
   - GPU 型号
   - 完整的错误信息
