# Workflow CPU-safe CI 调整

日期：2026-03-13

## 变更内容

- 将主线 CI 从无效的 CUDA matrix 构建，收敛为 `Format Check` 与 `Docs Build` 两个 Hosted Runner 可执行检查
- 保留 `push`、`pull_request`、`workflow_dispatch` 触发，继续覆盖文档站与源码格式质量
- 使用仓库现有 `package-lock.json` 构建 HonKit 文档，避免主线 CI 再被 GPU 依赖拖垮

## 背景

该仓库原先在 GitHub Hosted Runner 上保留了多个被停用的 CUDA job，主线验收价值有限。本次调整将 CI 聚焦到真实可跑且对仓库有价值的 CPU-safe 检查。
