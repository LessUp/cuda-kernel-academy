# Workflow 深度标准化

日期：2026-03-10

## 变更内容

- Pages workflow 重命名：`docs.yml` → `pages.yml`
- CI workflow 统一 `permissions: contents: read` 与 `concurrency` 配置
- Pages workflow 补充 `actions/configure-pages@v5` 步骤
- Pages workflow 添加 `paths` 触发过滤，减少无效构建

## 背景

全仓库第二轮 GitHub Actions 深度标准化：统一命名、权限、并发、路径过滤与缓存策略。
