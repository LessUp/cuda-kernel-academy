# GitHub Pages 部署指南

本文档介绍 CUDA Kernel Academy 文档站点的构建和部署流程。

## 快速开始

### 本地开发

```bash
# 安装依赖
npm install

# 启动本地服务器
npm run docs:serve

# 构建文档
npm run docs:build

# 完整构建（包含后处理优化）
npm run docs:build-prod
```

## 项目结构

```
.
├── book.json              # HonKit 配置文件
├── package.json           # Node.js 依赖
├── SUMMARY.md             # 文档目录结构
├── assets/                # 静态资源
│   ├── styles/           # 自定义 CSS
│   └── images/           # 图片和图标
├── scripts/              # 构建脚本
│   └── post-build.js     # 后处理脚本
├── docs/                 # 全局文档
├── _site/                # 构建输出目录
└── .github/workflows/    # CI/CD 配置
    ├── pages.yml         # 部署工作流
    └── ci.yml            # 持续集成
```

## 主要优化项

### 1. SEO 优化

- **Sitemap**: 自动生成 `sitemap.xml`，便于搜索引擎索引
- **Robots.txt**: 指导搜索引擎爬虫行为
- **Meta 标签**: 包含描述、关键词、作者信息
- **Open Graph**: 支持社交媒体分享卡片
- **Twitter Card**: Twitter 分享优化
- **结构化数据**: JSON-LD 格式的 Schema.org 标记

### 2. 用户体验优化

- **代码高亮**: Prism.js 语法高亮，支持 CUDA、C++、Python 等
- **代码复制**: 一键复制代码块
- **锚点链接**: 标题自动生成锚点，便于分享
- **编辑链接**: 每页右上角显示"编辑此页"链接
- **自定义样式**: 品牌配色、优化排版
- **404 页面**: 友好的错误页面，包含搜索功能
- **GitHub Star 按钮**: 侧边栏显示项目 Star 数

### 3. 性能优化

- **依赖缓存**: GitHub Actions 中缓存 `node_modules`
- **增量构建**: 只构建变更的文档
- **并行处理**: CI 任务并行运行

### 4. 搜索优化

- **Search Plus**: 增强版搜索插件，支持中文分词
- **搜索结果预览**: 显示匹配的文本片段

## 配置文件说明

### book.json

主要配置项：

```json
{
  "plugins": [
    "search-plus",          // 增强搜索
    "github-buttons",       // GitHub 按钮
    "edit-link",            // 编辑链接
    "anchorjs",             // 锚点生成
    "prism",                // 代码高亮
    "copy-code-button",     // 复制代码按钮
    "alerts"                // 警告提示框
  ]
}
```

### 后处理脚本 (scripts/post-build.js)

构建完成后自动执行：

1. 生成 `sitemap.xml`
2. 生成 `robots.txt`
3. 添加 SEO Meta 标签
4. 生成 `404.html`
5. 注入自定义 CSS

## 自定义样式

自定义样式位于 `assets/styles/website.css`，包含：

- 品牌配色方案 (NVIDIA 绿色 #76B900)
- 优化的字体栈（支持中文）
- 代码块样式（深色主题）
- 表格、引用、警告框样式
- 响应式布局优化
- 暗色模式支持 (prefers-color-scheme)

## CI/CD 流程

### 自动部署触发条件

以下文件变更会触发自动部署：
- `docs/**`
- `*/README.md`
- `*/docs/**`
- `SUMMARY.md`
- `book.json`
- `assets/**`
- `scripts/**`

### 工作流说明

1. **CI 工作流** (`.github/workflows/ci.yml`):
   - 代码格式检查
   - 文档构建验证
   - 链接检查
   - CMake 预设验证

2. **Pages 工作流** (`.github/workflows/pages.yml`):
   - 构建文档站点
   - 后处理优化
   - 部署到 GitHub Pages

## 故障排除

### 构建失败

```bash
# 清理并重新安装依赖
rm -rf node_modules package-lock.json
npm install

# 重新构建
npm run docs:build-prod
```

### 插件不工作

检查 `book.json` 中的插件配置是否正确，某些插件可能需要额外的配置。

### 样式未生效

确保 `assets/styles/website.css` 存在，并且 post-build 脚本正确执行。

## 性能监控

部署后可通过以下工具检查站点性能：

- [Google PageSpeed Insights](https://pagespeed.web.dev/)
- [GTmetrix](https://gtmetrix.com/)
- [WebPageTest](https://www.webpagetest.org/)

## 相关链接

- [HonKit 文档](https://honkit.netlify.app/)
- [GitHub Pages 文档](https://docs.github.com/pages)
- [GitBook 插件列表](https://github.com/honkit/honkit/wiki/Plugins)

## 更新日志

查看 [CHANGELOG.md](../CHANGELOG.md) 了解文档站点的更新历史。
