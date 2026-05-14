<script setup lang="ts">
import { ref } from 'vue'

interface Module {
  id: string
  name: string
  description: string
  link: string
  features: string[]
}

const modules: Module[] = [
  {
    id: 'sgemm',
    name: '01-SGEMM Tutorial',
    description: 'SGEMM optimization from naive to Tensor Core',
    link: '/zh/modules/01-sgemm',
    features: ['Naive', 'Tiled', 'Bank Conflict', 'Double Buffer', 'Tensor Core']
  },
  {
    id: 'tensorcraft',
    name: '02-TensorCraft Core',
    description: 'Production-grade CUDA operator library',
    link: '/zh/modules/02-tensorcraft',
    features: ['Header-only', 'Multi-arch', 'Python bindings', 'Error handling']
  },
  {
    id: 'hpc',
    name: '03-HPC Advanced',
    description: 'Advanced optimization patterns and CUDA features',
    link: '/zh/modules/03-hpc',
    features: ['FlashAttention', 'CUTLASS', 'CUDA 13', 'Quantization']
  },
  {
    id: 'inference',
    name: '04-Inference Engine',
    description: 'End-to-end inference system',
    link: '/zh/modules/04-inference',
    features: ['Memory Pool', 'Stream Manager', 'AutoTuner', 'Profiler']
  }
]

const selectedModule = ref<Module | null>(null)
const hoveredModule = ref<string | null>(null)

const selectModule = (module: Module) => {
  selectedModule.value = module
}
</script>

<template>
  <div class="system-architecture">
    <h4 class="section-title">System Architecture</h4>

    <!-- Architecture diagram -->
    <div class="architecture-diagram">
      <!-- Module nodes -->
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

      <!-- Dependency arrows (simplified with CSS) -->
      <div class="dependencies">
        <svg class="dependency-lines" viewBox="0 0 400 200">
          <!-- SGEMM → TensorCraft -->
          <path
            d="M 100 50 Q 150 50 200 100"
            fill="none"
            stroke="#76B900"
            stroke-width="2"
            stroke-dasharray="5,5"
            class="dep-line"
          />
          <!-- TensorCraft → HPC -->
          <path
            d="M 200 100 Q 250 50 300 50"
            fill="none"
            stroke="#76B900"
            stroke-width="2"
            stroke-dasharray="5,5"
            class="dep-line"
          />
          <!-- TensorCraft → Inference -->
          <path
            d="M 200 100 Q 250 150 300 150"
            fill="none"
            stroke="#76B900"
            stroke-width="2"
            stroke-dasharray="5,5"
            class="dep-line"
          />
          <!-- HPC → Inference -->
          <path
            d="M 300 50 Q 300 100 300 150"
            fill="none"
            stroke="#76B900"
            stroke-width="2"
            stroke-dasharray="5,5"
            class="dep-line"
          />
        </svg>
      </div>
    </div>

    <!-- Module details panel -->
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

      <a :href="selectedModule.link" class="details-link">
        View Documentation →
      </a>
    </div>

    <!-- Instructions -->
    <p v-else class="instructions">
      Click on a module to see details
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
  border-color: #76B900;
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(118, 185, 0, 0.15);
}

.module-node.selected {
  border-color: #76B900;
  background: rgba(118, 185, 0, 0.1);
}

.module-header {
  margin-bottom: 8px;
}

.module-id {
  font-family: var(--vp-font-family-mono);
  font-size: 0.75rem;
  font-weight: 700;
  color: #76B900;
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
  border: 1px solid #76B900;
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
  color: #76B900;
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
  background: rgba(118, 185, 0, 0.1);
  border-radius: 4px;
  font-size: 0.8rem;
  color: #76B900;
}

.details-link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: #76B900;
  font-weight: 600;
  text-decoration: none;
}

.details-link:hover {
  color: #8ce600;
}

.instructions {
  text-align: center;
  color: var(--vp-c-text-3);
  font-size: 0.9rem;
}
</style>
