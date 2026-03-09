# 2026-03-10 GitHub Pages 优化

## 变更

### GitHub Pages 工作流优化 (`pages.yml`)
- **Sparse-checkout**：仅检出文档相关文件（docs/、各子项目 README.md 和 docs/），跳过所有 CUDA/C++ 源码、测试、基准测试等大目录，显著减少 CI 检出时间和磁盘占用。
- **npm 缓存**：`actions/setup-node` 启用 `cache: npm`，后续构建复用 node_modules 缓存，加速 honkit 安装。
- **锁定 HonKit 版本**：新增 `package.json` + `package-lock.json`，将 honkit 固定为 `^3.7.1`，使用 `npm ci` 替代 `npm install`，确保可复现构建。
- **补全 paths 触发条件**：新增 `README.zh-CN.md`、`.gitbook.yaml`、`package.json`、`package-lock.json` 触发路径。

### Docs 徽章修复
- 修复 `README.zh-CN.md` 中 Docs 徽章链接：`docs.yml` → `pages.yml`，与实际工作流文件名一致。
- 在 `README.md`（英文）中新增 Docs 徽章，链接到 GitHub Pages 站点。

### 新增文件
- `.gitbook.yaml` — 显式声明 HonKit 首页为 `README.md`、目录为 `SUMMARY.md`。
- `package.json` — 声明 honkit 开发依赖，提供 `docs:build` / `docs:serve` 脚本。
- `package-lock.json` — 锁定依赖版本，支持 npm 缓存和可复现安装。

## 原因

- 原工作流每次全量检出整个仓库（含 4 个子项目的 CUDA 源码、测试、Docker 配置等），仅为构建 Markdown 文档，浪费 CI 资源。
- `npm install honkit` 无版本锁定，每次构建可能安装不同版本，存在不可复现风险。
- Docs 徽章引用了不存在的 `docs.yml`，导致徽章状态始终显示异常。
- 缺少 `.gitbook.yaml` 显式配置文件，HonKit 依赖隐式默认值。
