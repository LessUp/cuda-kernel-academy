<script setup lang="ts">
import { ref, computed } from 'vue'
import { useData, withBase } from 'vitepress'

interface Module {
  id: string
  name: string
  description: string
  link: string
  features: string[]
}

const { lang } = useData()
const isZh = computed(() => lang.value?.startsWith('zh'))

const modules = computed<Module[]>(() => {
  if (isZh.value) {
    return [
      {
        id: 'sgemm',
        name: '01-SGEMM 教程',
        description: '从 naive 到 Tensor Core 的 SGEMM 优化阶梯。',
        link: '/zh/modules/01-sgemm',
        features: ['Naive', 'Tiled', 'Bank Conflict', 'Double Buffer', 'Tensor Core']
      },
      {
        id: 'tensorcraft',
        name: '02-TensorCraft Core',
        description: '把教程 kernel 沉淀成可复用的 CUDA 算子库。',
        link: '/zh/modules/02-tensorcraft',
        features: ['Header-only', 'Multi-arch', 'Python bindings', 'Error handling']
      },
      {
        id: 'hpc',
        name: '03-HPC 进阶',
        description: '继续推进到更高级的优化模式和 CUDA 新特性。',
        link: '/zh/modules/03-hpc',
        features: ['FlashAttention', 'CUTLASS', 'CUDA 12+', 'Quantization']
      },
      {
        id: 'inference',
        name: '04-Inference Engine',
        description: '把优化后的 kernel 接入端到端推理系统。',
        link: '/zh/modules/04-inference',
        features: ['Memory Pool', 'Stream Manager', 'AutoTuner', 'Profiler']
      }
    ]
  }

  return [
    {
      id: 'sgemm',
      name: '01-SGEMM Tutorial',
      description: 'The kernel optimization ladder from naive SGEMM to Tensor Core variants.',
      link: '/en/modules/01-sgemm',
      features: ['Naive', 'Tiled', 'Bank Conflict', 'Double Buffer', 'Tensor Core']
    },
    {
      id: 'tensorcraft',
      name: '02-TensorCraft Core',
      description: 'How tutorial kernels become reusable CUDA operator building blocks.',
      link: '/en/modules/02-tensorcraft',
      features: ['Header-only', 'Multi-arch', 'Python bindings', 'Error handling']
    },
    {
      id: 'hpc',
      name: '03-HPC Advanced',
      description: 'Where the repo explores more advanced optimization patterns and newer CUDA features.',
      link: '/en/modules/03-hpc',
      features: ['FlashAttention', 'CUTLASS', 'CUDA 12+', 'Quantization']
    },
    {
      id: 'inference',
      name: '04-Inference Engine',
      description: 'How optimized kernels are integrated into an end-to-end inference runtime.',
      link: '/en/modules/04-inference',
      features: ['Memory Pool', 'Stream Manager', 'AutoTuner', 'Profiler']
    }
  ]
})

const ui = computed(() => isZh.value
  ? {
      sectionTitle: '模块关系图',
      viewDocs: '查看文档 →',
      clickHint: '点击模块查看说明'
    }
  : {
      sectionTitle: 'System Architecture',
      viewDocs: 'View documentation →',
      clickHint: 'Click a module to see details'
    }
)

const selectedModule = ref<Module | null>(null)
const hoveredModule = ref<string | null>(null)

const selectModule = (module: Module) => {
  selectedModule.value = module
}
</script>

<template>
  <div class="system-architecture">
    <h4 class="section-title">{{ ui.sectionTitle }}</h4>

    <div class="architecture-diagram">
      <div class="modules-container">
        <div
          v-for="module in modules"
          :key="module.id"
          class="module-node"
          :class="{
            hovered: hoveredModule === module.id,
            selected: selectedModule?.id === module.id
          }"
          @mouseenter="hoveredModule = module.id"
          @mouseleave="hoveredModule = null"
          @click="selectModule(module)"
        >
          <div class="module-header">
            <span class="module-id">{{ module.id.toUpperCase() }}</span>
          </div>
          <div class="module-name">{{ module.name }}</div>
        </div>
      </div>

      <div class="dependencies">
        <svg class="dependency-lines" viewBox="0 0 400 200">
          <path
            d="M 100 50 Q 150 50 200 100"
            fill="none"
            stroke="var(--vp-c-brand-1)"
            stroke-width="2"
            stroke-dasharray="5,5"
            class="dep-line"
          />
          <path
            d="M 200 100 Q 250 50 300 50"
            fill="none"
            stroke="var(--vp-c-brand-1)"
            stroke-width="2"
            stroke-dasharray="5,5"
            class="dep-line"
          />
          <path
            d="M 200 100 Q 250 150 300 150"
            fill="none"
            stroke="var(--vp-c-brand-1)"
            stroke-width="2"
            stroke-dasharray="5,5"
            class="dep-line"
          />
          <path
            d="M 300 50 Q 300 100 300 150"
            fill="none"
            stroke="var(--vp-c-brand-1)"
            stroke-width="2"
            stroke-dasharray="5,5"
            class="dep-line"
          />
        </svg>
      </div>
    </div>

    <div v-if="selectedModule" class="module-details">
      <div class="details-header">
        <h5>{{ selectedModule.name }}</h5>
        <button class="close-btn" @click="selectedModule = null">×</button>
      </div>

      <p class="details-description">{{ selectedModule.description }}</p>

      <div class="details-features">
        <span
          v-for="feature in selectedModule.features"
          :key="feature"
          class="feature-tag"
        >
          {{ feature }}
        </span>
      </div>

      <a :href="withBase(selectedModule.link)" class="details-link">
        {{ ui.viewDocs }}
      </a>
    </div>

    <p v-else class="instructions">
      {{ ui.clickHint }}
    </p>
  </div>
</template>

<style scoped>
.system-architecture {
  margin: 24px 0;
  padding: 24px;
  background: var(--vp-c-bg-alt);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
}

.section-title {
  margin: 0 0 20px 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.architecture-diagram {
  position: relative;
  height: 200px;
  margin-bottom: 20px;
}

.modules-container {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  padding: 0 20px;
}

.module-node {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 16px;
  background: var(--vp-c-bg);
  border: 2px solid var(--vp-c-divider);
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s;
}

.module-node:hover,
.module-node.hovered {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(118, 185, 0, 0.15);
}

.module-node.selected {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}

.module-header {
  margin-bottom: 8px;
}

.module-id {
  font-family: var(--vp-font-family-mono);
  font-size: 0.75rem;
  font-weight: 700;
  color: var(--vp-c-brand-1);
}

.module-name {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  text-align: center;
}

.dependencies {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.dependency-lines {
  width: 100%;
  height: 100%;
}

.dep-line {
  animation: dash 20s linear infinite;
}

@keyframes dash {
  to {
    stroke-dashoffset: -100;
  }
}

.module-details {
  padding: 20px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-brand-1);
  border-radius: 8px;
}

.details-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.details-header h5 {
  margin: 0;
  font-size: 1rem;
  color: var(--vp-c-brand-1);
}

.close-btn {
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: none;
  border: 1px solid var(--vp-c-divider);
  border-radius: 50%;
  cursor: pointer;
  font-size: 1.2rem;
  color: var(--vp-c-text-2);
}

.close-btn:hover {
  border-color: #f85149;
  color: #f85149;
}

.details-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  line-height: 1.6;
}

.details-features {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 16px;
}

.feature-tag {
  padding: 4px 12px;
  background: var(--vp-c-brand-soft);
  border-radius: 4px;
  font-size: 0.8rem;
  color: var(--vp-c-brand-1);
}

.details-link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: var(--vp-c-brand-1);
  font-weight: 600;
  text-decoration: none;
}

.details-link:hover {
  color: var(--vp-c-brand-2);
}

.instructions {
  text-align: center;
  color: var(--vp-c-text-3);
  font-size: 0.9rem;
}
</style>
