<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, computed } from 'vue'
import { useData } from 'vitepress'
import * as echarts from 'echarts'

interface DataItem {
  name: string
  value: number
  color?: string
}

const props = withDefaults(defineProps<{
  title?: string
  data?: DataItem[]
  unit?: string
  type?: 'bar' | 'line'
}>(), {
  title: '',
  data: () => [],
  unit: 'TFLOPS',
  type: 'bar'
})

const { isDark } = useData()
const chartRef = ref<HTMLDivElement | null>(null)
let chartInstance: echarts.ECharts | null = null

const axisColor = computed(() => isDark.value ? '#8b949e' : '#5b6472')
const tooltipBg = computed(() => isDark.value ? '#161b22' : '#ffffff')
const tooltipBorder = computed(() => isDark.value ? '#30363d' : '#d0d7de')
const textColor = computed(() => isDark.value ? '#e6edf3' : '#1f2937')

function getOption(): echarts.EChartsOption {
  const names = props.data.map(d => d.name)
  const values = props.data.map(d => d.value)
  const isCuBLAS = (name: string) => name.toLowerCase().includes('cublas')

  const seriesData = props.data.map((d, i) => {
    if (d.color) {
      return {
        value: d.value,
        itemStyle: {
          color: d.color,
          shadowBlur: 10,
          shadowColor: 'rgba(255, 255, 255, 0.4)'
        }
      }
    }
    if (isCuBLAS(d.name)) {
      return {
        value: d.value,
        itemStyle: {
          color: '#ffffff',
          shadowBlur: 12,
          shadowColor: 'rgba(255, 255, 255, 0.5)'
        }
      }
    }
    return {
      value: d.value,
      itemStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: '#a8ff00' },
          { offset: 1, color: '#76B900' }
        ])
      }
    }
  })

  return {
    backgroundColor: 'transparent',
    title: {
      text: props.title,
      left: 'center',
      top: 12,
      textStyle: {
        color: textColor.value,
        fontSize: 16,
        fontWeight: 600,
        fontFamily: 'Inter, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif'
      }
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: tooltipBg.value,
      borderColor: tooltipBorder.value,
      borderWidth: 1,
      textStyle: {
        color: textColor.value,
        fontSize: 13
      },
      formatter: (params: any) => {
        const p = Array.isArray(params) ? params[0] : params
        return `<div style="font-weight:600">${p.name}</div>` +
               `<div style="margin-top:4px">${p.value} ${props.unit}</div>`
      }
    },
    grid: {
      left: 60,
      right: 24,
      top: 60,
      bottom: 40
    },
    xAxis: {
      type: 'category',
      data: names,
      axisLine: { lineStyle: { color: '#30363d' } },
      axisTick: { show: false },
      axisLabel: {
        color: axisColor.value,
        fontSize: 12,
        interval: 0,
        rotate: names.length > 6 ? 30 : 0
      }
    },
    yAxis: {
      type: 'value',
      name: props.unit,
      nameTextStyle: {
        color: axisColor.value,
        fontSize: 12,
        padding: [0, 40, 0, 0]
      },
      axisLine: { show: false },
      axisTick: { show: false },
      splitLine: {
        lineStyle: {
          color: isDark.value ? 'rgba(48,54,61,0.6)' : 'rgba(208,215,222,0.6)',
          type: 'dashed'
        }
      },
      axisLabel: {
        color: axisColor.value,
        fontSize: 12
      }
    },
    series: [
      {
        type: props.type,
        data: seriesData,
        barMaxWidth: 40,
        itemStyle: {
          borderRadius: props.type === 'bar' ? [6, 6, 0, 0] : 0
        },
        lineStyle: {
          color: '#76B900',
          width: 3
        },
        areaStyle: props.type === 'line' ? {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(118,185,0,0.3)' },
            { offset: 1, color: 'rgba(118,185,0,0.02)' }
          ])
        } : undefined,
        symbol: 'circle',
        symbolSize: 8,
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(118,185,0,0.5)'
          }
        },
        animationDuration: 1000,
        animationEasing: 'cubicOut'
      }
    ]
  }
}

function initChart() {
  if (!chartRef.value) return
  chartInstance = echarts.init(chartRef.value, isDark.value ? 'dark' : undefined)
  chartInstance.setOption(getOption())
}

function resizeChart() {
  chartInstance?.resize()
}

onMounted(() => {
  initChart()
  window.addEventListener('resize', resizeChart)
})

onUnmounted(() => {
  window.removeEventListener('resize', resizeChart)
  chartInstance?.dispose()
  chartInstance = null
})

watch(() => [props.data, props.type, props.title, isDark.value], () => {
  chartInstance?.setOption(getOption(), true)
}, { deep: true })
</script>

<template>
  <div class="benchmark-chart-wrapper">
    <div ref="chartRef" class="benchmark-chart"></div>
  </div>
</template>

<style scoped>
.benchmark-chart-wrapper {
  background: var(--vp-c-bg-alt, #161b22);
  border: 1px solid var(--vp-c-border, #30363d);
  border-radius: 12px;
  padding: 16px;
  width: 100%;
}

.benchmark-chart {
  width: 100%;
  height: 360px;
}
</style>
