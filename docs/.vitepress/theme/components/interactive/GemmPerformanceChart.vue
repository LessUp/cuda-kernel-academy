<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useECharts, createGemmBarOption, gemmPerformanceData } from '../../composables/useECharts'

const chartContainer = ref<HTMLElement | null>(null)
const { chart, setOption } = useECharts(chartContainer)

const props = defineProps<{
  title?: string
  data?: typeof gemmPerformanceData
}>()

const chartData = props.data || gemmPerformanceData

onMounted(() => {
  setOption(createGemmBarOption(chartData))
})
</script>

<template>
  <div class="gemm-performance-chart">
    <h4 v-if="title" class="chart-title">{{ title }}</h4>
    <div ref="chartContainer" class="chart-container"></div>
  </div>
</template>

<style scoped>
.gemm-performance-chart {
  margin: 24px 0;
  padding: 20px;
  background: var(--vp-c-bg-alt);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
}

.chart-title {
  margin: 0 0 16px 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.chart-container {
  width: 100%;
  height: 400px;
}
</style>
