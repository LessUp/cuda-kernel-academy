# Design Document: Open Source Project Polish

## Overview

本设计文档描述了将 CUDA Kernel Academy 项目整理成规范开源项目的技术方案。设计遵循开源社区最佳实践，参考了 NVIDIA CUTLASS、PyTorch 等知名项目的组织方式。

## Architecture

### 项目文件结构

```
cuda-kernel-academy/
├── .github/                          # GitHub 配置
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml            # Bug 报告模板
│   │   ├── feature_request.yml       # 功能请求模板
│   │   └── config.yml                # Issue 模板选择器配置
│   ├── workflows/
│   │   └── ci.yml                    # 主 CI 流程
│   ├── PULL_REQUEST_TEMPLATE.md      # PR 模板
│   ├── dependabot.yml                # 依赖自动更新
│   └── markdown-link-check-config.json # Markdown 链接检查配置
├── docs/
│   ├── CODING_STYLE.md               # 已存在
│   ├── FAQ.md                        # 常见问题
│   ├── INSTALLATION.md               # 安装指南
│   ├── integration_examples.md       # 跨模块集成示例
│   ├── README.md                     # 文档索引页
│   └── TROUBLESHOOTING.md            # 故障排除
├── examples/                         # 示例目录
│   ├── 01_basic_gemm/
│   ├── 02_tensor_operations/
│   ├── CMakeLists.txt
│   └── README.md
├── changelog/                        # 变更记录目录
├── 01-sgemm-tutorial/                # 已存在
├── 02-tensorcraft-core/              # 已存在
├── 03-hpc-advanced/                  # 已存在
├── 04-inference-engine/              # 已存在
├── common/                           # 已存在
├── .clang-format                     # 已存在
├── .clang-tidy                       # 静态分析配置
├── .editorconfig                     # 编辑器统一配置
├── .gitattributes                    # Git 文件属性
├── .gitignore                        # 更新
├── .pre-commit-config.yaml           # Pre-commit 钩子
├── CMakeLists.txt                    # 根级 CMake
├── CMakePresets.json                 # CMake 预设
├── CHANGELOG.md                      # 变更日志
├── CODE_OF_CONDUCT.md                # 行为准则
├── CONTRIBUTING.md                   # 贡献指南
├── LICENSE                           # 已存在
├── README.md                         # 更新
├── SECURITY.md                       # 安全政策
└── VERSION                           # 版本号
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

#### Feature Request 模板

```yaml
# .github/ISSUE_TEMPLATE/feature_request.yml
name: Feature Request
description: Suggest an idea for CUDA Kernel Academy
labels: ["enhancement"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting an improvement!
  - type: textarea
    id: problem
    attributes:
      label: Problem Description
      description: Is your feature request related to a problem?
    validations:
      required: true
  - type: textarea
    id: solution
    attributes:
      label: Proposed Solution
      description: Describe the solution you'd like
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives Considered
      description: Any alternative solutions you've considered
  - type: dropdown
    id: module
    attributes:
      label: Related Module
      options:
        - 01-sgemm-tutorial
        - 02-tensorcraft-core
        - 03-hpc-advanced
        - 04-inference-engine
        - common
        - Other
```

#### Issue 模板选择器配置

```yaml
# .github/ISSUE_TEMPLATE/config.yml
blank_issues_enabled: false
contact_links:
  - name: 💬 Discussions
    url: https://github.com/LessUp/cuda-kernel-academy/discussions
    about: Ask questions and discuss ideas
  - name: 📚 Documentation
    url: https://github.com/LessUp/cuda-kernel-academy/tree/main/docs
    about: Read the documentation
```

#### Markdown 链接检查配置

```json
{
  "ignorePatterns": [
    { "pattern": "^https://github.com/LessUp" },
    { "pattern": "^mailto:" }
  ],
  "replacementPatterns": [],
  "httpHeaders": [
    {
      "urls": ["https://github.com"],
      "headers": { "Accept-Encoding": "zstd, br, gzip, deflate" }
    }
  ]
}
```

#### Dependabot 配置

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
      - "github-actions"
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
            "displayName": "Hopper (sm_90)",
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
Please report security vulnerabilities via GitHub's private vulnerability reporting feature.
Do NOT create public issues for security vulnerabilities.

## Response Timeline
- Acknowledgment: within 48 hours
- Initial assessment: within 1 week
- Fix release: depends on severity
```

#### docs/README.md

文档索引页，提供所有文档的导航入口：

- 链接到 INSTALLATION.md、FAQ.md、TROUBLESHOOTING.md
- 链接到 CODING_STYLE.md、integration_examples.md
- 链接到根目录的 CONTRIBUTING.md、CHANGELOG.md

#### docs/integration_examples.md

跨模块集成示例，展示如何将各子项目组合使用：

- tensorcraft-core 与 inference-engine 的集成
- 算子融合示例
- Python 绑定调用示例

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

All notable changes to CUDA Kernel Academy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

## [1.0.0] - 2026-01-07

### Added

#### Project Structure
- Comprehensive 4-module learning path for CUDA kernel development
- Unified project organization with shared common utilities

#### 01-sgemm-tutorial
- 5 SGEMM optimization levels: Naive → Tiled → Bank-Free → Double Buffer → Tensor Core
- Benchmark suite comparing all implementations against cuBLAS

#### 02-tensorcraft-core
- Header-only high-performance kernel library (GEMM, Attention, Conv, Norm, Sparse)
- Python bindings via pybind11

#### 03-hpc-advanced
- 7-step GEMM optimization path with Register Tiling and Software Pipelining
- CUDA 13 features: TMA, Thread Block Clusters, FP8
- Property-based testing with RapidCheck

#### 04-inference-engine
- Complete inference engine framework with Memory Pool, Stream Manager, Auto-tuner
- INT8/FP16 quantization, MNIST demo

[Unreleased]: https://github.com/LessUp/cuda-kernel-academy/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/LessUp/cuda-kernel-academy/releases/tag/v1.0.0
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

由于本设计主要涉及项目配置文件和文档，而非可执行代码，大部分需求属于结构性要求而非可测试的功能属性。以下是可验证的属性：

### Property 1: CI 构建成功

*For any* valid code commit to the main branch, the CI system SHALL successfully complete the build process without errors.

**Validates: Requirement 3.2, 3.3**

### Property 2: 格式检查一致性

*For any* source file in the repository, running clang-format SHALL produce no changes if the file is already properly formatted.

**Validates: Requirement 7.3, 7.5**

### Property 3: 示例可编译性

*For any* example in the examples/ directory, the example SHALL compile successfully with the default CMake configuration.

**Validates: Requirement 8.2**

### Property 4: 文档链接有效性

*For any* Markdown file in the repository, all internal and external links SHALL resolve to valid targets.

**Validates: Requirement 5.7**

## Error Handling

### CI 失败处理

- 构建失败：CI 系统将阻止 PR 合并
- 测试失败：CI 系统将报告失败的测试用例
- 格式检查失败：CI 系统将提示需要运行 clang-format

### 文档验证

- 使用 markdownlint 检查 Markdown 格式
- 使用 markdown-link-check 验证文档中的链接有效性（配置文件：.github/markdown-link-check-config.json）

## Testing Strategy

### 文件存在性测试

验证所有必需文件存在：

- [ ] CONTRIBUTING.md
- [ ] CODE_OF_CONDUCT.md
- [ ] SECURITY.md
- [ ] .github/ISSUE_TEMPLATE/bug_report.yml
- [ ] .github/ISSUE_TEMPLATE/feature_request.yml
- [ ] .github/ISSUE_TEMPLATE/config.yml
- [ ] .github/PULL_REQUEST_TEMPLATE.md
- [ ] .github/workflows/ci.yml
- [ ] .github/dependabot.yml
- [ ] .github/markdown-link-check-config.json
- [ ] CMakeLists.txt (根目录)
- [ ] CMakePresets.json
- [ ] CHANGELOG.md
- [ ] VERSION
- [ ] .clang-tidy
- [ ] .editorconfig
- [ ] .pre-commit-config.yaml
- [ ] .gitattributes
- [ ] docs/README.md
- [ ] docs/INSTALLATION.md
- [ ] docs/TROUBLESHOOTING.md
- [ ] docs/FAQ.md
- [ ] docs/integration_examples.md
- [ ] examples/README.md
- [ ] examples/CMakeLists.txt

### CI 集成测试

- 在 GitHub Actions 中验证构建流程
- 测试多架构构建 (sm_70, sm_80, sm_86, sm_90)
- 验证格式检查工作流

### 示例编译测试

- 每个示例独立编译测试
- 验证示例可以正常运行
