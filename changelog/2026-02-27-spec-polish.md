# 2026-02-27 Spec 文档优化

## 目的

优化 `.kiro/specs/open-source-project-polish/` 下的三个 spec 文件（requirements.md、design.md、tasks.md），使其与项目实际状态一致，消除三文档间的交叉引用错误和遗漏。

## 变更概览

### requirements.md

- **Glossary 补充**：新增 `Build_System`（Requirement 6 使用）和 `Example` 术语定义
- **Requirement 2**：新增 2.2（config.yml Issue 模板选择器）、原 2.3-2.4 顺延为 2.3-2.5，细化 bug report 需包含 CUDA 版本和 GPU 型号
- **Requirement 3**：新增 3.6（dependabot.yml 依赖自动更新）
- **Requirement 4**：细化 4.3（CHANGELOG 应包含 per-module release notes）和 4.4（明确 MAJOR.MINOR.PATCH 格式）
- **Requirement 5**：新增 5.4（docs/README.md 文档索引）、5.5（docs/integration_examples.md 跨模块示例）、5.7（markdown-link-check 链接检查），原 5.4 顺延为 5.6
- **Requirement 7**：新增 7.5（保持已有 .clang-format 配置）

### design.md

- **项目结构图**：移除不存在的 `release.yml` 和 `FUNDING.yml`；补充遗漏的 `config.yml`、`markdown-link-check-config.json`、`docs/integration_examples.md`、`docs/README.md`、`changelog/`、`examples/CMakeLists.txt`
- **组件设计补充**：新增 Issue 模板选择器（config.yml）、Markdown 链接检查配置、Dependabot 配置、docs/README.md、docs/integration_examples.md 的设计说明
- **技术修正**：Hopper preset 注释从 "CUDA 13" 改为 "sm_90"；SECURITY.md 占位符邮箱改为 GitHub 私密漏洞报告机制
- **CHANGELOG 示例**：从过度简化的 5 行更新为贴近实际的 per-module 详细格式
- **Correctness Properties**：新增 Property 4（文档链接有效性），修正 Property 1-2 的 Requirement 引用
- **Testing Strategy**：文件存在性测试清单从 15 项扩展为 22 项，覆盖所有实际文件

### tasks.md

- **新增任务**：4.3（markdown-link-check-config.json）、7.4（docs/README.md）、7.5（docs/integration_examples.md）
- **Requirement 交叉引用修正**：
  - Task 3.1: 2.1, 2.3 → 2.1, 2.4
  - Task 3.3: 2.2, 2.4 → 2.3, 2.5
  - Task 3.4: 2.1 → 2.2
  - Task 4.2: 3.1 → 3.6
  - Task 8.1: 3.5, 5.4 → 3.5, 5.6
- **Notes 更新**：补充已有文件免修改说明、changelog/ 用途、文档语言风格说明
- **Final Checkpoint**：新增 requirement 覆盖率验证步骤

### 最终验证修正

- **design.md**：移除 `markdown-link-check-config.json` JSON 示例中的非法 `//` 注释（JSON 不支持注释）
- **design.md**：补充 `feature_request.yml` 的完整 YAML 设计示例（之前仅有 `bug_report.yml` 的示例）

## 项目文件修正（基于 spec 验证）

在对 22 个文件进行全量审核后，修正了以下实际项目文件问题：

### SECURITY.md

- 移除占位符邮箱 `[INSERT SECURITY EMAIL]`，仅保留 GitHub 私密漏洞报告作为唯一报告渠道

### GitHub 用户名统一

- 将所有文件中的 `OWNER`、`yourusername`、`ORIGINAL_OWNER` 占位符统一替换为实际用户名 `LessUp`
- 涉及文件：README.md、SECURITY.md、CONTRIBUTING.md、CHANGELOG.md、CMakeLists.txt、docs/FAQ.md、docs/INSTALLATION.md、docs/TROUBLESHOOTING.md、docs/integration_examples.md、.github/ISSUE_TEMPLATE/config.yml、.github/markdown-link-check-config.json、04-inference-engine/docs/QUICK_START.md、04-inference-engine/docs/CONTRIBUTING.md、.kiro/specs/open-source-project-polish/design.md
- 合并 markdown-link-check-config.json 中重复的忽略模式

### README.md

- 更新项目结构图：添加 examples/、docs/ 下新增文件、.github/、CMakeLists.txt、CMakePresets.json、CONTRIBUTING.md、CHANGELOG.md
- 贡献指南部分添加 CONTRIBUTING.md 引用链接

### docs/README.md

- 修正误导性引用："各模块的 CONTRIBUTING.md" → 指向根目录的 CONTRIBUTING.md

## 影响

- 三文档间的交叉引用完全一致（38 条 Acceptance Criteria 全部有 Task 覆盖）
- 所有实际存在的项目文件均被 spec 覆盖
- 不存在的幽灵条目已清除
- design.md 中的代码示例语法正确
- 所有 GitHub URL 统一指向 `https://github.com/LessUp/cuda-kernel-academy`
- SECURITY.md 不再包含未填充的占位符
