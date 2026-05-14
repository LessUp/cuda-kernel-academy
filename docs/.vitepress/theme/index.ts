import DefaultTheme from 'vitepress/theme'
import './style.css'
import type { Theme } from 'vitepress'

// Existing components
import HeroSection from './components/HeroSection.vue'
import BenchmarkChart from './components/BenchmarkChart.vue'
import OptimizationLadder from './components/OptimizationLadder.vue'
import FlashAttentionViz from './components/FlashAttentionViz.vue'
import CudaTimeline from './components/CudaTimeline.vue'
import FeatureCard from './components/FeatureCard.vue'
import MetricBadge from './components/MetricBadge.vue'
import ReferenceBlock from './components/ReferenceBlock.vue'

// New interactive components
import GemmPerformanceChart from './components/interactive/GemmPerformanceChart.vue'
import GemmTilingInteractive from './components/interactive/GemmTilingInteractive.vue'
import PerformanceComparisonInteractive from './components/interactive/PerformanceComparisonInteractive.vue'
import FlashAttentionVizInteractive from './components/interactive/FlashAttentionVizInteractive.vue'
import KernelMemoryVisualization from './components/interactive/KernelMemoryVisualization.vue'

// New layout components
import SystemArchitectureDiagram from './components/layout/SystemArchitectureDiagram.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    // Existing components
    app.component('HeroSection', HeroSection)
    app.component('BenchmarkChart', BenchmarkChart)
    app.component('OptimizationLadder', OptimizationLadder)
    app.component('FlashAttentionViz', FlashAttentionViz)
    app.component('CudaTimeline', CudaTimeline)
    app.component('FeatureCard', FeatureCard)
    app.component('MetricBadge', MetricBadge)
    app.component('ReferenceBlock', ReferenceBlock)

    // New interactive components
    app.component('GemmPerformanceChart', GemmPerformanceChart)
    app.component('GemmTilingInteractive', GemmTilingInteractive)
    app.component('PerformanceComparisonInteractive', PerformanceComparisonInteractive)
    app.component('FlashAttentionVizInteractive', FlashAttentionVizInteractive)
    app.component('KernelMemoryVisualization', KernelMemoryVisualization)

    // New layout components
    app.component('SystemArchitectureDiagram', SystemArchitectureDiagram)
  }
} satisfies Theme
