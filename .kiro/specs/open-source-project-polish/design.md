# Design Document: Open Source Project Polish

## Overview

本设计文档描述了将 CUDA Kernel Academy 项目整理成规范开源项目的技术方案。设计遵循开源社区最佳实践，参考了 NVIDIA CUTLASS、PyTorch 等知名项目的组织方式。

## Architecture

### 项目文件结构

```
cuda-kernel-academy/
├── .github/                          # GitHub 配置
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   └── feature_request.yml
│   ├── workflows/
│   │   ├── ci.yml                    # 主 CI 流程
│   │   └── release.yml               # 发布流程
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── FUNDING.yml                   # 赞助配置 (可选)
│   └── dependabot.yml                # 依赖更新
├── docs/
│   ├── INSTALLATION.md               # 安装指南
│   ├── TROUBLESHOOTING.md            # 故障排除
│   ├── FAQ.md                        # 常见问题
│   ├── CODING_STYLE.md               # 已存在
│   └── README.md                     # 已存在
├── examples/                         # 新增示例目录
│   ├── 01_basic_gemm/
│   ├── 02_tensor_operations/
│   └── README.md
├── 01-sgemm-tutorial/                # 已存在
├── 02-tensorcraft-core/              # 已存在
├── 03-hpc-advanced/                  # 已存在
├── 04-inference-engine/              # 已存在
├── common/                           # 已存在
├── .clang-format                     # 已存在
├── .clang-tidy                       # 新增
├── .editorconfig                     # 新增
├── .gitattributes                    # 新增
├── .gitignore                        # 更新
├── .pre-commit-config.yaml           # 新增
├── CMakeLists.txt                    # 新增根级 CMake
├── CMakePresets.json                 # 新增
├── CHANGELOG.md                      # 新增
├── CODE_OF_CONDUCT.md                # 新增
├── CONTRIBUTING.md                   # 新增
├── LICENSE                           # 已存在
├── README.md                         # 更新
├── SECURITY.md                       # 新增
└── VERSION                           # 新增
```

## Components and Interfaces

### 1. GitHub 配置组件

#### Issue 模板 (YAML 格式)

```yaml
# .github/ISSUE_TEMPLATE/bug_report.yml
name: Bug Report
description: Report a bug in CUDA Kernel Academy
labels: ["bug", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  - type: input
    id: cuda-version
    attributes:
      label: CUDA Version
      placeholder: "e.g., 12.0"
    validations:
      required: true
  - type: input
    id: gpu-model
    attributes:
      label: GPU Model
      placeholder: "e.g., RTX 3090"
    validations:
      required: true
  - type: textarea
    id: description
    attributes:
      label: Bug Description
      description: A clear description of the bug
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Steps to Reproduce
      description: Steps to reproduce the behavior
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: Expected Behavior
      description: What you expected to happen
  - type: textarea
    id: logs
    attributes:
      label: Relevant Logs
      render: shell
```

#### PR 模板

```markdown
## Description
<!-- Describe your changes -->

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-reviewed the code
- [ ] Added tests for new functionality
- [ ] All tests pass locally
- [ ] Updated documentation if needed

## Related Issues
<!-- Link related issues: Fixes #123 -->
```

### 2. CI/CD 组件

#### GitHub Actions 工作流

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:12.0-devel-ubuntu22.04
    strategy:
      matrix:
        cuda_arch: [70, 80, 86, 90]
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: |
          apt-get update
          apt-get install -y cmake ninja-build
      - name: Configure
        run: |
          cmake -B build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_ARCHITECTURES=${{ matrix.cuda_arch }}
      - name: Build
        run: cmake --build build
      - name: Test
        run: ctest --test-dir build --output-on-failure

  format-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check formatting
        uses: jidicula/clang-format-action@v4.11.0
        with:
          clang-format-version: '17'
```

### 3. 构建系统组件

#### 根级 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(CUDAKernelAcademy 
    VERSION 1.0.0
    LANGUAGES CXX CUDA
    DESCRIPTION "Comprehensive CUDA kernel optimization learning project"
)

# Options
option(BUILD_SGEMM_TUTORIAL "Build 01-sgemm-tutorial" ON)
option(BUILD_TENSORCRAFT "Build 02-tensorcraft-core" ON)
option(BUILD_HPC_ADVANCED "Build 03-hpc-advanced" ON)
option(BUILD_INFERENCE_ENGINE "Build 04-inference-engine" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA Architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

# Common include
add_subdirectory(common)

# Subprojects
if(BUILD_TENSORCRAFT)
    add_subdirectory(02-tensorcraft-core)
endif()

if(BUILD_HPC_ADVANCED)
    add_subdirectory(03-hpc-advanced)
endif()

if(BUILD_INFERENCE_ENGINE)
    add_subdirectory(04-inference-engine)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Install
include(GNUInstallDirs)
install(DIRECTORY common/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
```

#### CMake Presets

```json
{
    "version": 6,
    "cmakeMinimumRequired": {"major": 3, "minor": 20, "patch": 0},
    "configurePresets": [
        {
            "name": "default",
            "displayName": "Default Config",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/build",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_CUDA_ARCHITECTURES": "70;80;86"
            }
        },
        {
            "name": "debug",
            "displayName": "Debug Config",
            "inherits": "default",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Debug"
            }
        },
        {
            "name": "ampere",
            "displayName": "Ampere Only",
            "inherits": "default",
            "cacheVariables": {
                "CMAKE_CUDA_ARCHITECTURES": "80;86"
            }
        },
        {
            "name": "hopper",
            "displayName": "Hopper (CUDA 13)",
            "inherits": "default",
            "cacheVariables": {
                "CMAKE_CUDA_ARCHITECTURES": "90"
            }
        }
    ],
    "buildPresets": [
        {
            "name": "default",
            "configurePreset": "default"
        },
        {
            "name": "debug",
            "configurePreset": "debug"
        }
    ]
}
```

### 4. 代码质量工具组件

#### .clang-tidy 配置

```yaml
Checks: >
  -*,
  bugprone-*,
  clang-analyzer-*,
  cppcoreguidelines-*,
  modernize-*,
  performance-*,
  readability-*,
  -modernize-use-trailing-return-type,
  -readability-magic-numbers,
  -cppcoreguidelines-avoid-magic-numbers

WarningsAsErrors: ''
HeaderFilterRegex: '.*'
FormatStyle: file
```

#### .editorconfig

```ini
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 4

[*.{yml,yaml}]
indent_size = 2

[*.md]
trim_trailing_whitespace = false

[Makefile]
indent_style = tab
```

#### Pre-commit 配置

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/pre-commit/mirrors-clang-format
    rev: v17.0.6
    hooks:
      - id: clang-format
        types_or: [c++, cuda]
```

### 5. 社区文件组件

#### CONTRIBUTING.md 结构

```markdown
# Contributing to CUDA Kernel Academy

## Getting Started
- Fork the repository
- Set up development environment
- Create a feature branch

## Development Workflow
1. Make changes
2. Run tests locally
3. Format code
4. Submit PR

## Code Style
- Follow .clang-format
- Use CUDA error checking macros
- Document public APIs

## Commit Messages
- Use conventional commits format
- feat: new feature
- fix: bug fix
- docs: documentation
- perf: performance improvement

## Pull Request Process
1. Update documentation
2. Add tests
3. Ensure CI passes
4. Request review
```

#### CODE_OF_CONDUCT.md

基于 Contributor Covenant v2.1，包含：
- 行为准则
- 执行标准
- 报告机制

#### SECURITY.md

```markdown
# Security Policy

## Supported Versions
| Version | Supported |
|---------|-----------|
| 1.x     | ✅        |

## Reporting a Vulnerability
Please report security vulnerabilities to: security@example.com
Do NOT create public issues for security vulnerabilities.
```

## Data Models

### 版本管理

```
VERSION 文件格式:
1.0.0

语义版本规则:
- MAJOR: 不兼容的 API 变更
- MINOR: 向后兼容的功能添加
- PATCH: 向后兼容的 bug 修复
```

### CHANGELOG 格式

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-01-07

### Added
- Initial release with 4 learning modules
- Comprehensive GEMM optimization tutorials
- TensorCraft core library
- HPC advanced features
- Inference engine framework
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

由于本设计主要涉及项目配置文件和文档，而非可执行代码，大部分需求属于结构性要求而非可测试的功能属性。以下是可验证的属性：

### Property 1: CI 构建成功

*For any* valid code commit to the main branch, the CI system SHALL successfully complete the build process without errors.

**Validates: Requirements 3.2**

### Property 2: 格式检查一致性

*For any* source file in the repository, running clang-format SHALL produce no changes if the file is already properly formatted.

**Validates: Requirements 7.3**

### Property 3: 示例可编译性

*For any* example in the examples/ directory, the example SHALL compile successfully with the default CMake configuration.

**Validates: Requirements 8.2**

## Error Handling

### CI 失败处理

- 构建失败：CI 系统将阻止 PR 合并
- 测试失败：CI 系统将报告失败的测试用例
- 格式检查失败：CI 系统将提示需要运行 clang-format

### 文档验证

- 使用 markdownlint 检查 Markdown 格式
- 使用 link checker 验证文档中的链接有效性

## Testing Strategy

### 文件存在性测试

验证所有必需文件存在：
- [ ] CONTRIBUTING.md
- [ ] CODE_OF_CONDUCT.md
- [ ] SECURITY.md
- [ ] .github/ISSUE_TEMPLATE/bug_report.yml
- [ ] .github/ISSUE_TEMPLATE/feature_request.yml
- [ ] .github/PULL_REQUEST_TEMPLATE.md
- [ ] .github/workflows/ci.yml
- [ ] CMakeLists.txt (根目录)
- [ ] CMakePresets.json
- [ ] CHANGELOG.md
- [ ] VERSION
- [ ] .clang-tidy
- [ ] .editorconfig
- [ ] .pre-commit-config.yaml
- [ ] .gitattributes

### CI 集成测试

- 在 GitHub Actions 中验证构建流程
- 测试多架构构建 (sm_70, sm_80, sm_86, sm_90)
- 验证格式检查工作流

### 示例编译测试

- 每个示例独立编译测试
- 验证示例可以正常运行

