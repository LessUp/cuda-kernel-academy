import DefaultTheme from 'vitepress/theme'
import HeroSection from './components/HeroSection.vue'
import BenchmarkChart from './components/BenchmarkChart.vue'
import OptimizationLadder from './components/OptimizationLadder.vue'
import FlashAttentionViz from './components/FlashAttentionViz.vue'
import CudaTimeline from './components/CudaTimeline.vue'
import FeatureCard from './components/FeatureCard.vue'
import MetricBadge from './components/MetricBadge.vue'
import ReferenceBlock from './components/ReferenceBlock.vue'
import './style.css'
import type { Theme } from 'vitepress'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('HeroSection', HeroSection)
    app.component('BenchmarkChart', BenchmarkChart)
    app.component('OptimizationLadder', OptimizationLadder)
    app.component('FlashAttentionViz', FlashAttentionViz)
    app.component('CudaTimeline', CudaTimeline)
    app.component('FeatureCard', FeatureCard)
    app.component('MetricBadge', MetricBadge)
    app.component('ReferenceBlock', ReferenceBlock)
  }
} satisfies Theme
