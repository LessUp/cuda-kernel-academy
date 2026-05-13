# Proposal: VitePress Migration with NVIDIA Theme

## Problem

CUDA Kernel Academy 的 GitHub Pages 目前基于 **HonKit**（GitBook 的旧 fork），存在以下致命缺陷：

1. **技术栈老化**：HonKit 社区已停滞，无现代文档特性（Mermaid 图表、本地搜索、LLMs.txt、深色模式自动切换）
2. **视觉平庸**：白色背景 + 简单绿色点缀，毫无 NVIDIA 的科技感和专业度，无法在众多 CUDA 学习项目中脱颖而出
3. **内容表现力弱**：ASCII 文本图无法直观展示 kernel 优化路径、内存访问模式、性能演进
4. **无学术规范**：缺少参考文献系统，文档像笔记而非工程/学术成果
5. **SEO/AI 不可见**：无 sitemap、无 og:meta、无 LLMs.txt，搜索引擎和 AI 助手难以索引

## Solution

将文档站从 HonKit 激进迁移到 **VitePress**（kimi-cli 同款技术栈），并建立：

- **NVIDIA 风格深色主题**：Dark Mode First，绿色强调，CSS Grid 装饰，科技感十足
- **Mermaid 图表**：所有架构图、优化路径、数据流图全部可视化
- **ECharts 交互图表**：Benchmark Dashboard、性能对比柱状图/折线图、精度切换
- **学术引用体系**：每篇技术文档末尾附 References 区块，仿学术论文格式
- **LLMs.txt + SEO**：让 AI 搜索引擎（Perplexity、ChatGPT、Claude 等）能精准索引本站

## Scope

| In Scope | Out of Scope |
|---|---|
| 删除 HonKit 全部基础设施 | 修改 C++/CUDA 源码 |
| 建立 VitePress 骨架 + 自定义主题 | 修改 CMake 构建系统 |
| 4 模块文档内容迁移 + 重构 | 新增 CUDA kernel 算法 |
| Mermaid 图表重绘 | 改动 GitHub Actions CI 测试 |
| ECharts Vue 组件开发 | 数据库/后端服务 |
| 中英双语结构 | |
| CI/CD 部署更新 | |
| SEO + LLMs.txt | |

## Success Criteria

1. `npm run docs:build` 零报错，输出 `docs/.vitepress/dist/`
2. 首页视觉惊艳，深色主题正确渲染，NVIDIA Green (#76B900) 贯穿全站
3. 所有 Mermaid 图表在构建后正常显示
4. Benchmark Dashboard 包含交互式 ECharts 图表
5. 每篇技术文档有 References 区块
6. 生成 `sitemap.xml` 和 `.well-known/llms.txt`
7. GitHub Actions Pages 部署成功
