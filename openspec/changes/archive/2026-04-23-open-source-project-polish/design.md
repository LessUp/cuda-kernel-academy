# Design: Open Source Project Polish

## Context

### Background

CUDA Kernel Academy 是一个多模块 CUDA 教育项目，包含 4 个学习模块：
- `01-sgemm-tutorial/` - SGEMM 优化基础 (Standalone Makefile)
- `02-tensorcraft-core/` - Header-only 内核库 (CMake)
- `03-hpc-advanced/` - 高级 CUDA 技术 (CMake, C++20 required)
- `04-inference-engine/` - 深度学习推理引擎 (CMake)

项目参考了 NVIDIA CUTLASS、PyTorch 等知名项目的组织方式。

### Current State

项目已具备核心 CUDA 代码和基础文档，但缺乏开源项目必需的配套文件：
- 无社区贡献指南
- 无标准化 Issue/PR 模板
- 无自动化 CI/CD
- 构建系统分散

## Goals

1. **社区友好**: 建立完整的社区贡献支持体系
2. **自动化**: 配置自动化 CI/CD 流水线保障代码质量
3. **文档完善**: 完善项目文档体系，降低学习门槛
4. **构建统一**: 统一构建系统，支持模块化构建

## Non-Goals

- 不涉及核心 CUDA 代码的修改
- 不改变现有模块架构
- 不修改现有 API 接口

## Decisions

### Decision 1: 采用 YAML 格式 Issue 模板

**理由**: YAML 格式提供更好的表单体验，支持字段验证和条件逻辑。相比 Markdown 模板，YAML 模板可以引导用户提供结构化信息，减少无效 Issue。

**实现**:
- `bug_report.yml` - 包含 CUDA 版本、GPU 型号、复现步骤等必填字段
- `feature_request.yml` - 包含问题描述、建议方案、模块关联等字段
- `config.yml` - 禁用空白 Issue，引导到 Discussions

### Decision 2: 使用 GitHub Actions 作为 CI 平台

**理由**: GitHub Actions 与 GitHub 深度集成，配置简单，支持自托管 Runner。使用 NVIDIA 官方 CUDA Docker 镜像确保环境一致性。

**实现**:
- 使用 `nvidia/cuda:12.0-devel-ubuntu22.04` 作为构建环境
- 多架构矩阵构建 (sm_70, sm_80, sm_86, sm_90)
- 格式检查使用 `clang-format-action`

### Decision 3: 根级 CMake 统一构建

**理由**: 统一的 CMake 配置简化了构建流程，同时保留各模块的独立构建能力。通过 CMake 选项支持选择性构建。

**实现**:
```cmake
option(BUILD_SGEMM_TUTORIAL "Build 01-sgemm-tutorial" ON)
option(BUILD_TENSORCRAFT "Build 02-tensorcraft-core" ON)
# ... 其他模块选项
```

### Decision 4: Pre-commit Hooks 本地质量保障

**理由**: Pre-commit hooks 在提交前自动运行格式检查和验证，减少 CI 反馈周期，提升开发效率。

**实现**:
- clang-format 代码格式检查
- trailing-whitespace 清理
- YAML/JSON 语法验证

## Risks and Trade-offs

### Risk 1: CI 在无 GPU 环境下无法运行 CUDA 测试

**概率**: 高
**影响**: 中

**缓解措施**:
- CI 仅运行 CPU 安全检查 (格式、文档、配置验证)
- CUDA 测试需在本地 GPU 环境运行
- 文档明确说明本地测试要求

### Risk 2: 模块 01 使用独立 Makefile

**概率**: 已确定
**影响**: 低

**缓解措施**:
- 根级 CMake 不包含模块 01
- 文档明确说明模块 01 的独立构建方式
- 保持 Makefile 与项目风格一致

### Trade-off 1: 多架构构建增加 CI 时间

**决策**: 接受额外 CI 时间换取广泛 GPU 兼容性验证

**影响**: 每次 PR 检查时间约 10-15 分钟

### Trade-off 2: 双语文档维护成本

**决策**: 保持中文为主、英文技术术语保留的风格

**影响**: 需要双语维护，但符合目标用户群体习惯

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

## Correctness Properties

### Property 1: CI 构建成功

*For any* valid code commit to the main branch, the CI system SHALL successfully complete the build process without errors.

**Validates**: Requirement 3.2, 3.3

### Property 2: 格式检查一致性

*For any* source file in the repository, running clang-format SHALL produce no changes if the file is already properly formatted.

**Validates**: Requirement 7.3, 7.5

### Property 3: 示例可编译性

*For any* example in the examples/ directory, the example SHALL compile successfully with the default CMake configuration.

**Validates**: Requirement 8.2

### Property 4: 文档链接有效性

*For any* Markdown file in the repository, all internal and external links SHALL resolve to valid targets.

**Validates**: Requirement 5.7

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
