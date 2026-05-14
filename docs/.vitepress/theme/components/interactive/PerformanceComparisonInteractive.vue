<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useECharts, gemmPerformanceData } from '../../composables/useECharts'

const chartContainer = ref<HTMLElement | null>(null)
const { chart, setOption } = useECharts(chartContainer)

const props = defineProps<{
  title?: string
}>()

onMounted(() => {
  setOption({
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' }
    },
    legend: {
      data: ['GFLOPS', '% of cuBLAS'],
      top: 0
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: gemmPerformanceData.versions,
      axisLabel: {
        rotate: 20,
        fontSize: 11
      }
    },
    yAxis: [
      {
        type: 'value',
        name: 'GFLOPS',
        position: 'left',
        axisLine: {
          show: true,
          lineStyle: { color: '#76B900' }
        },
        axisLabel: { color: '#76B900' }
      },
      {
        type: 'value',
        name: '% of cuBLAS',
        position: 'right',
        max: 100,
        axisLine: {
          show: true,
          lineStyle: { color: '#8ce600' }
        },
        axisLabel: { color: '#8ce600' }
      }
    ],
    series: [
      {
        name: 'GFLOPS',
        type: 'bar',
        yAxisIndex: 0,
        data: gemmPerformanceData.gflops,
        itemStyle: {
          color: '#76B900',
          borderRadius: [4, 4, 0, 0]
        },
        label: {
          show: true,
          position: 'top',
          formatter: '{c}',
          color: '#e6edf3',
          fontSize: 11
        }
      },
      {
        name: '% of cuBLAS',
        type: 'line',
        yAxisIndex: 1,
        data: gemmPerformanceData.percentage,
        smooth: true,
        symbol: 'circle',
        symbolSize: 8,
        lineStyle: {
          width: 3,
          color: '#8ce600'
        },
        itemStyle: { color: '#8ce600' }
      }
    ]
  })
})
</script>

<template>
  <div class="performance-comparison">
    <h4 v-if="title" class="chart-title">{{ title }}</h4>
    <div ref="chartContainer" class="chart-container"></div>
  </div>
</template>

<style scoped>
.performance-comparison {
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
