# 贡献指南

感谢你对 Mini-Inference Engine 的兴趣！本文档说明如何为项目做出贡献。

## 行为准则

- 尊重所有贡献者
- 保持专业和建设性的讨论
- 接受建设性的批评

## 如何贡献

### 报告 Bug

1. 检查是否已有相关 issue
2. 创建新 issue，包含:
   - 问题描述
   - 复现步骤
   - 预期行为
   - 实际行为
   - 环境信息 (GPU, CUDA 版本, OS)

### 提交功能请求

1. 描述功能需求
2. 说明使用场景
3. 提供可能的实现方案

### 提交代码

#### 1. Fork 仓库

```bash
git clone https://github.com/yourusername/mini-inference-engine.git
cd mini-inference-engine
git remote add upstream https://github.com/original/mini-inference-engine.git
```

#### 2. 创建分支

```bash
git checkout -b feature/my-feature
# 或
git checkout -b fix/bug-description
```

#### 3. 编写代码

遵循代码规范 (见下文)

#### 4. 测试

```bash
# 构建
mkdir build && cd build
cmake ..
make -j$(nproc)

# 运行测试
./tests

# 运行 benchmark 验证性能
./benchmark
```

#### 5. 提交

```bash
git add .
git commit -m "feat: add new feature"
# 或
git commit -m "fix: fix bug description"
```

提交信息格式:
- `feat:` 新功能
- `fix:` Bug 修复
- `docs:` 文档更新
- `refactor:` 代码重构
- `test:` 测试相关
- `perf:` 性能优化

#### 6. 推送并创建 PR

```bash
git push origin feature/my-feature
```

然后在 GitHub 上创建 Pull Request。

## 代码规范

### C++ 代码风格

```cpp
// 命名规范
class ClassName;           // 类名: PascalCase
void function_name();      // 函数名: snake_case
int variable_name;         // 变量名: snake_case
const int CONSTANT_NAME;   // 常量: UPPER_SNAKE_CASE
int member_variable_;      // 成员变量: snake_case + 下划线后缀

// 缩进: 4 空格
void function() {
    if (condition) {
        // code
    }
}

// 大括号: 同行
if (condition) {
    // code
} else {
    // code
}

// 注释
// 单行注释
/* 多行注释 */

/// Doxygen 文档注释
/// @param x 参数描述
/// @return 返回值描述
```

### CUDA 代码风格

```cpp
// Kernel 命名: snake_case
__global__ void my_kernel(...);

// 模板参数: UPPER_SNAKE_CASE
template<int BLOCK_SIZE, bool USE_SHARED>
__global__ void templated_kernel(...);

// 共享内存: 前缀 s_ 或后缀 _shared
__shared__ float s_data[256];
__shared__ float data_shared[256];

// 寄存器变量: 前缀 r_ 或后缀 _reg
float r_sum = 0.0f;
float sum_reg = 0.0f;
```

### 文件组织

```
include/
├── common.h          # 通用定义
├── kernels.cuh       # Kernel 声明
├── feature.h         # 功能模块头文件
└── ...

src/
├── kernel_name.cu    # Kernel 实现
├── feature.cpp       # 功能模块实现
└── ...

tests/
├── test_feature.cpp  # 功能测试
└── ...
```

### 头文件保护

```cpp
#pragma once

// 或者
#ifndef MINI_INFERENCE_FEATURE_H
#define MINI_INFERENCE_FEATURE_H

// ...

#endif // MINI_INFERENCE_FEATURE_H
```

### 错误处理

```cpp
// 使用 CUDA_CHECK 宏
CUDA_CHECK(cudaMalloc(&ptr, size));

// 使用异常
if (invalid_input) {
    throw std::invalid_argument("Invalid input: ...");
}

// 返回错误码 (可选)
bool function() {
    if (error) {
        return false;
    }
    return true;
}
```

## 测试要求

### 单元测试

每个新功能都需要测试:

```cpp
#include <gtest/gtest.h>
#include "feature.h"

class FeatureTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }
};

TEST_F(FeatureTest, BasicFunctionality) {
    // 测试基本功能
    EXPECT_EQ(expected, actual);
}

TEST_F(FeatureTest, EdgeCases) {
    // 测试边界情况
}

TEST_F(FeatureTest, ErrorHandling) {
    // 测试错误处理
    EXPECT_THROW(function_with_invalid_input(), std::invalid_argument);
}
```

### 性能测试

对于性能相关的更改:

```cpp
TEST_F(FeatureTest, Performance) {
    GpuTimer timer;
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        function_under_test();
    }
    
    // Benchmark
    timer.start();
    for (int i = 0; i < 20; i++) {
        function_under_test();
    }
    timer.stop();
    
    float avg_time = timer.elapsed_ms() / 20;
    printf("Average time: %.3f ms\n", avg_time);
    
    // 可选: 性能断言
    EXPECT_LT(avg_time, max_acceptable_time);
}
```

## 文档要求

### 代码注释

```cpp
/// @brief 执行优化的 GEMM 运算
/// @param A 输入矩阵 A (M x K)
/// @param B 输入矩阵 B (K x N)
/// @param C 输出矩阵 C (M x N)
/// @param M 矩阵 A 的行数
/// @param N 矩阵 B 的列数
/// @param K 矩阵 A 的列数 / 矩阵 B 的行数
/// @param stream CUDA 流 (可选)
void launch_optimized_gemm(const float* A, const float* B, float* C,
                           int M, int N, int K, cudaStream_t stream = 0);
```

### README 更新

如果添加新功能，更新 README.md:
- 功能列表
- 使用示例
- API 说明

### 文档文件

对于重要功能，添加或更新 docs/ 中的文档。

## 审查流程

1. 自动检查
   - 编译通过
   - 测试通过
   - 代码风格检查

2. 人工审查
   - 代码质量
   - 设计合理性
   - 文档完整性

3. 性能验证 (如适用)
   - 不引入性能回归
   - 新优化有效

## 发布流程

版本号遵循语义化版本:
- MAJOR: 不兼容的 API 更改
- MINOR: 向后兼容的功能添加
- PATCH: 向后兼容的 Bug 修复

## 联系方式

- GitHub Issues: 报告问题和功能请求
- Pull Requests: 代码贡献
- Discussions: 一般讨论

## 致谢

感谢所有贡献者的付出！
