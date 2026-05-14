<script setup lang="ts">
import { ref, computed } from 'vue'

// Animation state
const currentStep = ref(0)
const isPlaying = ref(false)
const animationSpeed = ref(1000) // ms per step

// FlashAttention steps
const steps = [
  {
    name: 'Load Q Block',
    description: 'Load Q tile (Br × d) into registers',
    memory: 'Q block: O(Br × d)'
  },
  {
    name: 'Load K,V Block',
    description: 'Load K and V tiles (Bc × d) into shared memory',
    memory: 'K,V blocks: O(Bc × d)'
  },
  {
    name: 'Compute QK^T',
    description: 'Compute attention scores S = Q × K^T / √d',
    memory: 'S block: O(Br × Bc)'
  },
  {
    name: 'Online Softmax',
    description: 'Update max and sum for numerical stability',
    memory: 'O(Br) scalars'
  },
  {
    name: 'Compute Output',
    description: 'O = softmax(S) × V',
    memory: 'Output block: O(Br × d)'
  },
  {
    name: 'Repeat',
    description: 'Process next K,V block (no O(N²) materialization)',
    memory: 'Total: O(N) memory'
  }
]

const currentStepData = computed(() => steps[currentStep.value])

const nextStep = () => {
  currentStep.value = (currentStep.value + 1) % steps.length
}

const prevStep = () => {
  currentStep.value = (currentStep.value - 1 + steps.length) % steps.length
}

const resetAnimation = () => {
  currentStep.value = 0
  isPlaying.value = false
}
</script>

<template>
  <div class="flash-attention-viz">
    <h4 class="section-title">FlashAttention Algorithm Visualization</h4>

    <!-- Step indicator -->
    <div class="step-indicator">
      <div
        v-for="(step, idx) in steps"
        :key="idx"
        class="step-dot"
        :class="{ active: idx === currentStep, completed: idx < currentStep }"
        @click="currentStep = idx"
      >
        {{ idx + 1 }}
      </div>
    </div>

    <!-- Current step visualization -->
    <div class="step-content">
      <div class="step-header">
        <span class="step-number">Step {{ currentStep + 1 }}/{{ steps.length }}</span>
        <h5 class="step-name">{{ currentStepData.name }}</h5>
      </div>

      <p class="step-description">{{ currentStepData.description }}</p>

      <div class="memory-info">
        <span class="memory-label">Memory:</span>
        <code class="memory-value">{{ currentStepData.memory }}</code>
      </div>
    </div>

    <!-- Controls -->
    <div class="controls">
      <button class="control-btn" @click="prevStep">← Prev</button>
      <button class="control-btn primary" @click="nextStep">Next →</button>
      <button class="control-btn" @click="resetAnimation">Reset</button>
    </div>

    <!-- Memory comparison -->
    <div class="memory-comparison">
      <div class="comparison-item">
        <span class="comparison-label">Standard Attention</span>
        <span class="comparison-value bad">O(N²) memory</span>
      </div>
      <div class="comparison-item">
        <span class="comparison-label">FlashAttention</span>
        <span class="comparison-value good">O(N) memory</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.flash-attention-viz {
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

.step-indicator {
  display: flex;
  justify-content: center;
  gap: 8px;
  margin-bottom: 24px;
}

.step-dot {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: var(--vp-c-bg);
  border: 2px solid var(--vp-c-divider);
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.step-dot:hover {
  border-color: var(--vp-c-brand-1);
}

.step-dot.active {
  background: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-bg);
}

.step-dot.completed {
  background: var(--vp-c-brand-soft);
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.step-content {
  padding: 20px;
  background: var(--vp-c-bg);
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
  margin-bottom: 16px;
}

.step-header {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 12px;
}

.step-number {
  font-size: 0.85rem;
  color: var(--vp-c-text-3);
}

.step-name {
  margin: 0;
  font-size: 1rem;
  color: var(--vp-c-brand-1);
}

.step-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  line-height: 1.6;
}

.memory-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.memory-label {
  font-size: 0.85rem;
  color: var(--vp-c-text-3);
}

.memory-value {
  padding: 4px 8px;
  background: var(--vp-c-brand-soft);
  border-radius: 4px;
  font-size: 0.85rem;
  color: var(--vp-c-brand-1);
}

.controls {
  display: flex;
  justify-content: center;
  gap: 12px;
  margin-bottom: 20px;
}

.control-btn {
  padding: 8px 20px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
  transition: all 0.2s;
}

.control-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.control-btn.primary {
  background: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-bg);
}

.control-btn.primary:hover {
  background: var(--vp-c-brand-2);
}

.memory-comparison {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.comparison-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 16px;
  background: var(--vp-c-bg);
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
}

.comparison-label {
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
}

.comparison-value {
  font-family: var(--vp-font-family-mono);
  font-size: 0.9rem;
  font-weight: 600;
}

.comparison-value.bad {
  color: #f85149;
}

.comparison-value.good {
  color: var(--vp-c-brand-1);
}
</style>
