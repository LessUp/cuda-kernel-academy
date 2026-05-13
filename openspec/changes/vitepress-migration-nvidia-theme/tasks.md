# Tasks: VitePress Migration with NVIDIA Theme

## Phase 1: Infrastructure (Prompt 1)

- [ ] Delete old HonKit files: book.json, SUMMARY.md, .gitbook.yaml, .bookignore, assets/styles/website.css, assets/images/favicon.svg, scripts/post-build.js
- [ ] Create docs/package.json with VitePress + Mermaid + LLMs dependencies
- [ ] Create docs/.vitepress/config.ts with zh/en locales, nav, sidebar, head meta
- [ ] Create docs/.vitepress/theme/index.ts (theme entry)
- [ ] Create docs/.vitepress/theme/Layout.vue (custom layout)
- [ ] Create docs/.vitepress/theme/style.css (NVIDIA dark theme CSS variables)
- [ ] Create docs/.vitepress/theme/custom.css (additional component styles)
- [ ] Create docs/public/favicon.svg
- [ ] Create docs/public/robots.txt
- [ ] Update root package.json (replace honkit scripts with VitePress)
- [ ] Update .gitignore (remove _site, add docs/.vitepress/dist and cache)
- [ ] Build test: `npm run docs:build` should produce dist/

## Phase 2: Content Migration (Prompt 2)

- [ ] Migrate README.md → docs/zh/index.md (hero homepage)
- [ ] Migrate docs/*.md → docs/zh/guides/ and docs/zh/reference/
- [ ] Migrate 01-sgemm-tutorial/README.md → docs/zh/modules/01-sgemm.md + 5 sub-pages
- [ ] Migrate 02-tensorcraft-core/docs/*.md → docs/zh/modules/02/
- [ ] Migrate 03-hpc-advanced/docs/*.md → docs/zh/modules/03/
- [ ] Migrate 04-inference-engine/docs/*.md → docs/zh/modules/04/
- [ ] Create docs/zh/benchmarks/index.md with placeholder performance table
- [ ] Create docs/zh/roadmap.md with Mermaid learning path
- [ ] Convert all ASCII diagrams to Mermaid charts
- [ ] Add ## References section to all technical docs with proper citations
- [ ] Create English mirror structure in docs/en/
- [ ] Fix all internal links to VitePress paths

## Phase 3: Interactive Components (Prompt 3)

- [ ] Create docs/.vitepress/theme/components/BenchmarkChart.vue (ECharts wrapper)
- [ ] Create docs/.vitepress/theme/components/OptimizationLadder.vue
- [ ] Create docs/.vitepress/theme/components/FlashAttentionViz.vue
- [ ] Create docs/.vitepress/theme/components/CudaTimeline.vue
- [ ] Create docs/.vitepress/theme/components/FeatureCard.vue
- [ ] Create docs/.vitepress/theme/components/MetricBadge.vue
- [ ] Integrate components into relevant markdown pages
- [ ] Implement FP32/FP16/INT8 toggle in BenchmarkChart
- [ ] Verify all components render correctly in dark mode
- [ ] Verify responsive behavior on mobile widths

## Phase 4: CI/CD & SEO (Prompt 4)

- [ ] Rewrite .github/workflows/pages.yml for VitePress build
- [ ] Configure VitePress head SEO meta tags (og:, twitter:, description)
- [ ] Enable VitePress built-in sitemap generation
- [ ] Configure vitepress-plugin-llms with project description
- [ ] Create docs/public/404.html with NVIDIA styling
- [ ] Update root README.md Pages badge and documentation links
- [ ] End-to-end build verification
- [ ] Preview and visual QA check

## Verification Checklist

- [ ] `npm run docs:build` exits 0
- [ ] `dist/index.html` exists and renders
- [ ] Dark mode is default, NVIDIA Green accent visible
- [ ] At least 5 Mermaid charts render correctly
- [ ] BenchmarkChart shows bars and toggle works
- [ ] All nav links resolve (no 404)
- [ ] Chinese and English homepages accessible
- [ ] Search returns results
- [ ] 404 page styled correctly
- [ ] GitHub Actions deploys successfully
