# Spec: VitePress Migration Technical Spec

## Tech Stack

| Layer | Technology | Version | Reason |
|---|---|---|---|
| SSG Framework | VitePress | ^1.5.0 | Vue team, Vite-based, fast HMR, built-in i18n/local search |
| Diagrams | Mermaid | ^11.12.2 | Architecture, flow, sequence, gantt charts |
| Mermaid Integration | vitepress-plugin-mermaid | ^2.0.17 | SSR-safe Mermaid rendering in VitePress |
| Charts | ECharts | ^5.5.0 | Interactive bar/line/sankey charts for benchmarks |
| LLM Discovery | vitepress-plugin-llms | ^1.10.0 | Auto-generate /.well-known/llms.txt |
| Runtime | Node.js | >= 22 | VitePress requirement |

## Theme Design System

### Color Palette (Dark Mode First)

```
Background Primary:   #0d1117  (page bg)
Background Secondary: #161b22  (cards, code blocks)
Background Tertiary:  #0f1f0a  (hero gradient end)
Border:               #30363d  (dividers, table borders)
Accent Primary:       #76B900  (NVIDIA Green - links, buttons, highlights)
Accent Hover:         #00E676  (brighter green on hover)
Accent Danger:        #ff6b6b  (warnings, important notes)
Text Primary:         #e6edf3  (headings, body)
Text Secondary:       #8b949e  (captions, metadata)
Text Muted:           #484f58  (disabled, borders)
```

### Typography

- English: Inter (UI), JetBrains Mono (code)
- Chinese: system-ui, "PingFang SC", "Noto Sans SC", sans-serif
- Base size: 16px, line-height 1.7
- Code blocks: 14px, border-radius 8px

### Spacing & Shape

- Card border-radius: 12px
- Button border-radius: 8px
- Section padding: 64px vertical on desktop, 32px on mobile
- Content max-width: 900px (reading comfort)
- Hero: full-width, min-height 60vh, CSS grid line decoration

## File Mapping (Old → New)

| Old Path | New Path | Action |
|---|---|---|
| `book.json` | `docs/.vitepress/config.ts` | Migrate config |
| `SUMMARY.md` | `docs/.vitepress/config.ts` sidebar | Programmatic nav |
| `.gitbook.yaml` | deleted | Not needed |
| `assets/styles/website.css` | `docs/.vitepress/theme/style.css` | Rewrite with CSS vars |
| `assets/images/favicon.svg` | `docs/public/favicon.svg` | Move |
| `scripts/post-build.js` | deleted | VitePress handles sitemap/meta |
| `README.md` | `docs/zh/index.md` + `docs/en/index.md` | Split to home pages |
| `docs/*.md` | `docs/zh/guides/*.md`, `docs/zh/reference/*.md` | Reorganize |
| `01-sgemm-tutorial/README.md` | `docs/zh/modules/01-sgemm.md` | Migrate + enhance |
| `02-tensorcraft-core/docs/*.md` | `docs/zh/modules/02/*.md` | Migrate + enhance |
| `03-hpc-advanced/docs/*.md` | `docs/zh/modules/03/*.md` | Migrate + enhance |
| `04-inference-engine/docs/*.md` | `docs/zh/modules/04/*.md` | Migrate + enhance |

## URL Routing

```
/                          → zh homepage
/zh/                       → zh homepage
/zh/guides/getting-started → 快速开始
/zh/modules/01-sgemm       → SGEMM 教程
/zh/modules/02-tensorcraft → TensorCraft Core
/zh/modules/03-hpc         → HPC Advanced
/zh/modules/04-inference  → Inference Engine
/zh/benchmarks/            → 性能基准测试
/zh/roadmap/             → 学习路线图
/zh/reference/           → 参考文档
/en/                     → English mirror
```

## Component Inventory

| Component | File | Purpose |
|---|---|---|
| `BenchmarkChart.vue` | `docs/.vitepress/theme/components/` | ECharts bar/line chart wrapper |
| `OptimizationLadder.vue` | `docs/.vitepress/theme/components/` | 7-step SGEMM optimization path |
| `FlashAttentionViz.vue` | `docs/.vitepress/theme/components/` | SRAM vs HBM comparison |
| `CudaTimeline.vue` | `docs/.vitepress/theme/components/` | CUDA version feature timeline |
| `FeatureCard.vue` | `docs/.vitepress/theme/components/` | Homepage feature grid cards |
| `MetricBadge.vue` | `docs/.vitepress/theme/components/` | Hero metric numbers |
| `ReferenceBlock.vue` | `docs/.vitepress/theme/components/` | Academic reference list styling |

## Reference Citation Format

```markdown
正文中的引用使用脚注语法：
FlashAttention 通过 Tiling 和 Online Softmax 减少 HBM 访问[^1]。

## References

[^1]: Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS* 2022. https://arxiv.org/abs/2205.14135
[^2]: NVIDIA. "CUDA C++ Programming Guide." v12.6, 2024. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
```

## Post-migration stabilization requirements

### Theme-adaptive diagrams

- Mermaid 图示内容必须避免硬编码会破坏浅色 / 深色模式切换的 `fill` / `color` 十六进制颜色。
- 文档构建路径必须包含针对该约束的静态回归测试。

### Academy-style content

- 中英双语的首页、架构白皮书、benchmark、roadmap 应提供：
  - 目标读者说明
  - 阅读顺序 / 学习路径
  - 模块之间的连接关系
  - benchmark 解释与边界
  - 对应论文或外部参考锚点

### Inference-engine build seam

- `04-inference-engine` 在父项目构建时必须优先消费稳定的 `TensorCraft::tensorcraft` target。
- 只有 standalone 构建场景才允许通过 sibling path 探测 `../02-tensorcraft-core`。
- 当父项目构建启用 TensorCraft 集成但上游 target 缺失时，配置阶段必须明确失败，而不是静默回退到 repo 布局探测。
