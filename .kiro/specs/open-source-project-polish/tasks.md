# Implementation Plan: Open Source Project Polish

## Overview

本实现计划将 CUDA Kernel Academy 项目整理成规范的开源项目。任务按照依赖关系和优先级排序，从基础配置文件开始，逐步完善社区文件、CI/CD 和文档。

## Tasks

- [x] 1. 创建项目元数据和配置文件
  - [x] 1.1 创建 VERSION 文件
    - 创建 `VERSION` 文件，内容为 `1.0.0`
    - _Requirements: 4.4_
  - [x] 1.2 创建 .gitattributes 文件
    - 配置文件类型处理规则 (LF 换行、二进制文件等)
    - _Requirements: 4.1_
  - [x] 1.3 更新 .gitignore 文件
    - 添加所有构建目录、IDE 配置、临时文件的忽略规则
    - _Requirements: 4.2_
  - [x] 1.4 创建 .editorconfig 文件
    - 配置统一的编辑器设置 (缩进、换行、字符集)
    - _Requirements: 7.2_
  - [x] 1.5 创建 .clang-tidy 配置文件
    - 配置静态分析检查规则
    - _Requirements: 7.1_
  - [x] 1.6 创建 .pre-commit-config.yaml 文件
    - 配置 pre-commit hooks (格式检查、trailing whitespace 等)
    - _Requirements: 7.4_

- [x] 2. 创建社区贡献文件
  - [x] 2.1 创建 CONTRIBUTING.md 文件
    - 包含开发环境设置、代码风格、提交规范、PR 流程
    - _Requirements: 1.1, 1.2_
  - [x] 2.2 创建 CODE_OF_CONDUCT.md 文件
    - 基于 Contributor Covenant v2.1
    - _Requirements: 1.3_
  - [x] 2.3 创建 SECURITY.md 文件
    - 包含支持版本表、漏洞报告流程
    - _Requirements: 1.4_
  - [x] 2.4 创建 CHANGELOG.md 文件
    - 遵循 Keep a Changelog 格式，记录 1.0.0 版本
    - _Requirements: 4.3_

- [x] 3. 创建 GitHub 模板文件
  - [x] 3.1 创建 .github/ISSUE_TEMPLATE/bug_report.yml
    - YAML 格式的 Bug 报告模板，包含 CUDA 版本、GPU 型号等字段
    - _Requirements: 2.1, 2.3_
  - [x] 3.2 创建 .github/ISSUE_TEMPLATE/feature_request.yml
    - YAML 格式的功能请求模板
    - _Requirements: 2.1_
  - [x] 3.3 创建 .github/PULL_REQUEST_TEMPLATE.md
    - 包含变更描述、类型选择、检查清单
    - _Requirements: 2.2, 2.4_
  - [x] 3.4 创建 .github/ISSUE_TEMPLATE/config.yml
    - 配置 Issue 模板选择器
    - _Requirements: 2.1_

- [x] 4. 创建 CI/CD 配置
  - [x] 4.1 创建 .github/workflows/ci.yml
    - 配置构建、测试、格式检查工作流
    - 支持多 CUDA 架构 (sm_70, sm_80, sm_86, sm_90)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 7.3_
  - [x] 4.2 创建 .github/dependabot.yml
    - 配置依赖自动更新
    - _Requirements: 3.1_

- [x] 5. Checkpoint - 验证配置文件
  - 确保所有配置文件语法正确
  - 验证 YAML 文件格式

- [x] 6. 创建根级构建系统
  - [x] 6.1 创建根目录 CMakeLists.txt
    - 配置项目元信息、C++/CUDA 标准
    - 添加子项目选项 (BUILD_SGEMM_TUTORIAL, BUILD_TENSORCRAFT 等)
    - 添加 common 和各子项目
    - 配置安装目标
    - _Requirements: 6.1, 6.2, 6.4_
  - [x] 6.2 创建 CMakePresets.json
    - 配置 default, debug, ampere, hopper 预设
    - _Requirements: 6.3_

- [x] 7. 创建文档文件
  - [x] 7.1 创建 docs/INSTALLATION.md
    - 详细的安装指南，覆盖 Linux、Docker 环境
    - 包含依赖要求、构建步骤、验证方法
    - _Requirements: 5.1_
  - [x] 7.2 创建 docs/TROUBLESHOOTING.md
    - 常见问题及解决方案
    - CUDA 版本兼容性、编译错误、运行时错误
    - _Requirements: 5.2_
  - [x] 7.3 创建 docs/FAQ.md
    - 常见问题解答
    - 项目定位、学习路径、贡献方式
    - _Requirements: 5.3_

- [x] 8. 更新主 README
  - [x] 8.1 更新 README.md 添加 CI 徽章
    - 添加 GitHub Actions 构建状态徽章
    - 确保现有徽章格式一致
    - _Requirements: 3.5, 5.4_

- [x] 9. 创建示例目录
  - [x] 9.1 创建 examples/README.md
    - 示例目录说明文档
    - _Requirements: 8.1_
  - [x] 9.2 创建 examples/01_basic_gemm/ 示例
    - 基础 GEMM 使用示例
    - 包含 CMakeLists.txt 和带注释的源代码
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  - [x] 9.3 创建 examples/02_tensor_operations/ 示例
    - Tensor 操作示例
    - 包含 CMakeLists.txt 和带注释的源代码
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  - [x] 9.4 创建 examples/CMakeLists.txt
    - 示例目录的 CMake 配置
    - _Requirements: 8.2_

- [x] 10. Final Checkpoint - 项目完整性验证
  - 确保所有文件已创建
  - 验证 CMake 配置语法正确
  - 检查文档链接有效性

## Notes

- 任务按依赖顺序排列，配置文件优先于社区文件
- CI 配置依赖于构建系统配置
- 示例代码依赖于核心库的存在
- 所有 Markdown 文件使用中英双语或纯中文，保持与现有文档风格一致

