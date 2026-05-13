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
