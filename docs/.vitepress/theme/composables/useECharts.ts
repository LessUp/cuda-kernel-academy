import { ref, onMounted, onUnmounted, watch, type Ref } from 'vue'
import * as echarts from 'echarts'

/**
 * ECharts composable for Vue 3
 * Handles chart initialization, resize, and cleanup
 */
export function useECharts(container: Ref<HTMLElement | null>) {
  const chart: Ref<echarts.ECharts | null> = ref(null)
  const isReady = ref(false)

  const initChart = () => {
    if (!container.value || chart.value) return

    chart.value = echarts.init(container.value, undefined, {
      renderer: 'canvas'
    })
    isReady.value = true

    // Apply NVIDIA theme colors
    chart.value.setOption({
      backgroundColor: 'transparent',
      textStyle: {
        fontFamily: 'Inter, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif'
      },
      color: ['#76B900', '#8ce600', '#00E676', '#79c0ff', '#ffa657', '#d2a8ff'],
      grid: {
        left: '10%',
        right: '10%',
        top: '15%',
        bottom: '15%'
      },
      xAxis: {
        axisLine: { lineStyle: { color: '#30363d' } },
        axisLabel: { color: '#8b949e' },
        splitLine: { lineStyle: { color: '#30363d', type: 'dashed' } }
      },
      yAxis: {
        axisLine: { lineStyle: { color: '#30363d' } },
        axisLabel: { color: '#8b949e' },
        splitLine: { lineStyle: { color: '#30363d', type: 'dashed' } }
      },
      legend: {
        textStyle: { color: '#8b949e' }
      },
      tooltip: {
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        textStyle: { color: '#e6edf3' }
      }
    })
  }

  const setOption = (option: echarts.EChartsOption, opts?: { notMerge?: boolean; lazyUpdate?: boolean }) => {
    if (chart.value) {
      chart.value.setOption(option, opts)
    }
  }

  const resize = () => {
    chart.value?.resize()
  }

  const dispose = () => {
    chart.value?.dispose()
    chart.value = null
    isReady.value = false
  }

  onMounted(() => {
    initChart()
    window.addEventListener('resize', resize)
  })

  onUnmounted(() => {
    window.removeEventListener('resize', resize)
    dispose()
  })

  return {
    chart,
    isReady,
    setOption,
    resize,
    dispose
  }
}

/**
 * Default GEMM performance data
 */
export const gemmPerformanceData = {
  versions: ['Naive', 'Tiled', 'Bank Free', 'Double Buffer', 'Register Block', 'Tensor Core'],
  gflops: [50, 200, 250, 300, 350, 800],
  percentage: [5, 20, 25, 30, 35, 80]
}

/**
 * Create bar chart option for GEMM performance
 */
export function createGemmBarOption(data: typeof gemmPerformanceData) {
  return {
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (params: any) => {
        const p = params[0]
        return `${p.name}<br/>Performance: ${p.value} GFLOPS<br/>vs cuBLAS: ${data.percentage[p.dataIndex]}%`
      }
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: data.versions,
      axisLabel: {
        rotate: 30,
        fontSize: 11
      }
    },
    yAxis: [
      {
        type: 'value',
        name: 'GFLOPS',
        nameTextStyle: { color: '#8b949e' }
      },
      {
        type: 'value',
        name: '% of cuBLAS',
        nameTextStyle: { color: '#8b949e' },
        max: 100
      }
    ],
    series: [
      {
        name: 'Performance',
        type: 'bar',
        data: data.gflops,
        itemStyle: {
          borderRadius: [4, 4, 0, 0]
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(118, 185, 0, 0.5)'
          }
        },
        label: {
          show: true,
          position: 'top',
          formatter: '{c}',
          color: '#e6edf3'
        }
      }
    ]
  }
}

/**
 * Create line chart option for optimization ladder
 */
export function createOptimizationLadderOption(data: typeof gemmPerformanceData) {
  return {
    tooltip: {
      trigger: 'axis'
    },
    xAxis: {
      type: 'category',
      data: data.versions,
      boundaryGap: false
    },
    yAxis: {
      type: 'value',
      name: 'GFLOPS'
    },
    series: [
      {
        name: 'Performance',
        type: 'line',
        data: data.gflops,
        smooth: true,
        symbol: 'circle',
        symbolSize: 10,
        lineStyle: {
          width: 3,
          color: '#76B900'
        },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(118, 185, 0, 0.4)' },
            { offset: 1, color: 'rgba(118, 185, 0, 0.05)' }
          ])
        },
        markLine: {
          data: [{ yAxis: 1000, name: 'cuBLAS baseline' }],
          lineStyle: { color: '#ffab00', type: 'dashed' },
          label: { formatter: 'cuBLAS baseline' }
        }
      }
    ]
  }
}
