<script setup lang="ts">
interface Step {
  id: number
  title: string
  bullets: string[]
  speedup: string
}

const steps: Step[] = [
  {
    id: 1,
    title: 'Naive Global Memory',
    bullets: ['每个线程读取 A/B 全局内存一次', '无共享内存复用', '内存带宽成为瓶颈'],
    speedup: 'Baseline'
  },
  {
    id: 2,
    title: 'Shared Memory Tiling',
    bullets: ['A/B 分块载入 Shared Memory', '线程协作加载，减少全局访问', '计算与访存比例提升'],
    speedup: '↑ 4×'
  },
  {
    id: 3,
    title: 'Coalesced Access',
    bullets: ['调整线程到数据的映射', '全局内存合并访问 (128B)', 'Shared Memory Bank Conflict 消除'],
    speedup: '↑ 6×'
  },
  {
    id: 4,
    title: 'Double Buffering',
    bullets: ['Software Pipeline 隐藏延迟', '预取下一块数据到寄存器', '计算与访存重叠'],
    speedup: '↑ 9×'
  },
  {
    id: 5,
    title: 'Vectorized Load',
    bullets: ['float4 / half2 向量加载', '减少指令数，提升吞吐', '寄存器压力可控'],
    speedup: '↑ 14×'
  },
  {
    id: 6,
    title: 'Tensor Core (WMMA)',
    bullets: ['Warp-level矩阵乘法指令', '混合精度 FP16/BF16', '峰值算力逼近理论值'],
    speedup: '↑ 18×'
  },
  {
    id: 7,
    title: 'cuBLAS (Reference)',
    bullets: ['NVIDIA 官方优化库', '自动调优与内核选择', '多版本架构适配'],
    speedup: '↑ 42×'
  }
]
</script>

<template>
  <div class="optimization-ladder">
    <div
      v-for="(step, index) in steps"
      :key="step.id"
      class="ladder-step"
    >
      <div class="step-card">
        <div class="step-header">
          <span class="step-number">{{ step.id }}</span>
          <span class="step-speedup">{{ step.speedup }}</span>
        </div>
        <h3 class="step-title">{{ step.title }}</h3>
        <ul class="step-bullets">
          <li v-for="(b, i) in step.bullets" :key="i">{{ b }}</li>
        </ul>
      </div>
      <div v-if="index < steps.length - 1" class="step-arrow" :style="{ '--arrow-opacity': 0.3 + (index / (steps.length - 1)) * 0.7 }">
        <span class="arrow-icon">→</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.optimization-ladder {
  display: flex;
  flex-wrap: wrap;
  align-items: stretch;
  gap: 0;
  margin: 24px 0;
}

.ladder-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1 1 0;
  min-width: 160px;
}

.step-card {
  background: var(--vp-c-bg-alt, #161b22);
  border: 1px solid var(--vp-c-border, #30363d);
  border-radius: 12px;
  padding: 20px;
  width: 100%;
  transition: transform 0.25s ease, box-shadow 0.25s ease;
  cursor: default;
}

.step-card:hover {
  transform: translateY(-6px);
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35), 0 0 0 1px rgba(118, 185, 0, 0.2);
  border-color: rgba(118, 185, 0, 0.5);
}

.step-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.step-number {
  font-size: 2rem;
  font-weight: 800;
  color: #76B900;
  line-height: 1;
  font-family: var(--vp-font-family-mono);
}

.step-speedup {
  font-size: 0.75rem;
  font-weight: 700;
  color: #0d1117;
  background: #76B900;
  padding: 4px 8px;
  border-radius: 999px;
}

.step-title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--vp-c-text-1, #e6edf3);
  margin: 0 0 10px 0;
}

.step-bullets {
  list-style: none;
  padding: 0;
  margin: 0;
}

.step-bullets li {
  position: relative;
  padding-left: 14px;
  font-size: 0.85rem;
  color: var(--vp-c-text-2, #8b949e);
  line-height: 1.5;
  margin-bottom: 6px;
}

.step-bullets li::before {
  content: '';
  position: absolute;
  left: 0;
  top: 7px;
  width: 6px;
  height: 6px;
  background: #76B900;
  border-radius: 50%;
}

.step-arrow {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px 0;
  width: 100%;
}

.arrow-icon {
  font-size: 1.25rem;
  color: #76B900;
  opacity: var(--arrow-opacity);
  font-weight: 700;
}

@media (max-width: 768px) {
  .optimization-ladder {
    flex-direction: column;
    gap: 0;
  }
  .ladder-step {
    min-width: 100%;
    flex-direction: row;
    align-items: center;
  }
  .step-card {
    flex: 1;
  }
  .step-arrow {
    width: auto;
    padding: 0 8px;
    transform: rotate(90deg);
  }
}
</style>
