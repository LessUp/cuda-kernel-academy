# Design: VitePress Migration Architecture

## Directory Tree (Post-Migration)

```
cuda-kernel-academy/
├── docs/
│   ├── .vitepress/
│   │   ├── config.ts           # Main VitePress config (locales, nav, sidebar, head)
│   │   ├── theme/
│   │   │   ├── index.ts        # Theme entry, component registration
│   │   │   ├── Layout.vue      # Custom layout wrapper (hero, footer)
│   │   │   ├── style.css       # Global CSS variables + overrides
│   │   │   ├── custom.css      # Additional component styles
│   │   │   └── components/
│   │   │       ├── BenchmarkChart.vue
│   │   │       ├── OptimizationLadder.vue
│   │   │       ├── FlashAttentionViz.vue
│   │   │       ├── CudaTimeline.vue
│   │   │       ├── FeatureCard.vue
│   │   │       ├── MetricBadge.vue
│   │   │       └── ReferenceBlock.vue
│   │   └── dist/               # Build output (gitignored)
│   ├── public/
│   │   ├── favicon.svg
│   │   ├── robots.txt
│   │   └── 404.html
│   ├── zh/
│   │   ├── index.md            # Homepage (hero + features)
│   │   ├── guides/
│   │   │   ├── getting-started.md
│   │   │   ├── workflow.md
│   │   │   └── installation.md
│   │   ├── modules/
│   │   │   ├── 01-sgemm.md
│   │   │   ├── 01/
│   │   │   │   ├── sgemm-naive.md
│   │   │   │   ├── sgemm-tiled.md
│   │   │   │   ├── sgemm-bank-conflict.md
│   │   │   │   ├── sgemm-double-buffer.md
│   │   │   │   └── sgemm-tensor-core.md
│   │   │   ├── 02-tensorcraft.md
│   │   │   ├── 02/
│   │   │   │   ├── architecture.md
│   │   │   │   ├── api-reference.md
│   │   │   │   └── optimization-guide.md
│   │   │   ├── 03-hpc.md
│   │   │   ├── 03/
│   │   │   │   ├── gemm-optimization.md
│   │   │   │   ├── memory-optimization.md
│   │   │   │   ├── reduction-optimization.md
│   │   │   │   ├── flash-attention.md
│   │   │   │   └── cuda13-features.md
│   │   │   ├── 04-inference.md
│   │   │   └── 04/
│   │   │       ├── architecture.md
│   │   │       ├── performance-tuning.md
│   │   │       └── quick-start.md
│   │   ├── benchmarks/
│   │   │   └── index.md
│   │   ├── roadmap.md
│   │   └── reference/
│   │       ├── coding-style.md
│   │       ├── troubleshooting.md
│   │       └── integration-examples.md
│   └── en/                     # Mirror of zh/ (English translations)
│       └── (same structure)
├── package.json                # Root package.json (VitePress scripts)
├── .github/
│   └── workflows/
│       └── pages.yml           # Updated for VitePress
└── (C++ source unchanged)
```

## Theme Implementation Strategy

### CSS Variable Override Map

VitePress exposes CSS custom properties under `:root` and `.dark`. We override:

```css
/* Dark mode (default) */
.dark {
  --vp-c-bg: #0d1117;
  --vp-c-bg-alt: #161b22;
  --vp-c-bg-elv: #1c2128;
  --vp-c-divider: #30363d;
  --vp-c-border: #30363d;
  --vp-c-text-1: #e6edf3;
  --vp-c-text-2: #8b949e;
  --vp-c-text-3: #484f58;
  --vp-c-brand-1: #76B900;
  --vp-c-brand-2: #8ce600;
  --vp-c-brand-3: #00E676;
  --vp-c-brand-soft: rgba(118, 185, 0, 0.15);
}
```

### Hero Section Design

- Full viewport width, 60vh min-height
- Background: `linear-gradient(135deg, #0d1117 0%, #1a2332 50%, #0f1f0a 100%)`
- Overlay: CSS `background-image: linear-gradient(...)` grid lines (1px rgba(118,185,0,0.08))
- Left: Logo (CKA monogram) + tagline
- Right: 3D floating card mockup (CSS-only, no WebGL)
- Bottom: 4 metric badges in a row

### Homepage Feature Grid

- 6 cards in 3x2 grid (responsive: 2 cols tablet, 1 col mobile)
- Card style: #161b22 bg, 1px #30363d border, 12px radius, hover lifts with shadow
- Each card: icon (48px, green), title, description, tag pills

## Component Data Flow

```
Markdown → VitePress Renderer → Vue Components → ECharts/Mermaid → DOM
     ↑                                    ↓
     └──── Frontmatter (YAML) ←──────────┘
```

Components read data from:
1. Frontmatter YAML (static config)
2. Props from markdown (dynamic data)
3. Component internal reactive state (user interactions)

## Build Pipeline

```
Source Markdown
      ↓
VitePress (Vite bundler)
      ↓
vitepress-plugin-mermaid (SSR-safe Mermaid render)
      ↓
vitepress-plugin-llms (generate .well-known/llms.txt)
      ↓
Vue SFC compilation (custom components)
      ↓
CSS post-processing (Vite CSS plugin)
      ↓
Static output: docs/.vitepress/dist/
      ↓
GitHub Actions → deploy-pages
```

## Post-Migration Stabilization

### 1. Theme-safe diagrams as a content invariant

Mermaid 图示的主题适配问题不再通过“页面作者手工记住颜色规范”来保证，而是通过两层保护：

1. **内容层约束**：文档中的 Mermaid fenced blocks 不再硬编码会破坏浅色模式的 `fill` / `color` 十六进制颜色。
2. **构建层回归测试**：`docs/tests/` 下的静态测试在 docs 构建前执行，阻止主题敌对的 Mermaid 样式重新进入仓库。

这让“图示可读性”成为构建时可验证的属性，而不是人工约定。

### 2. Academy-style landing pages

迁移后的站点不只保留页面结构，还要补齐高杠杆导读页：

- `docs/{en,zh}/index.md`
- `docs/{en,zh}/whitepaper/architecture.md`
- `docs/{en,zh}/benchmarks/index.md`
- `docs/{en,zh}/roadmap.md`

这些页面负责把仓库组织成：

- **入口**：谁应该先读哪里
- **架构地图**：模块之间如何连接
- **证据地图**：benchmark 数据如何解释、边界在哪里
- **学习地图**：每一阶段该读什么、配什么论文

实现上优先复用现有组件（如 `SystemArchitectureDiagram`、`BenchmarkChart`、`ReferenceBlock`），避免新增平行抽象。

### 3. Inference-engine dependency seam

`04-inference-engine` 的 TensorCraft 依赖分成两个明确场景：

1. **父项目构建**：必须消费上游已提供的 `TensorCraft::tensorcraft` target；如果缺失则明确失败。
2. **standalone 构建**：才允许尝试通过 `../02-tensorcraft-core` 做 sibling fallback；若不存在则退回本地 GEMM 实现。

这样做的目的不是增加灵活性，而是减少父项目场景对 repo 布局的耦合，并把可变性限制在真正需要的 standalone seam 上。
