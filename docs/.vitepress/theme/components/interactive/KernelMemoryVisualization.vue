<script setup lang="ts">
import { ref, computed } from 'vue'

// Memory hierarchy levels
const levels = [
  { name: 'HBM', size: '16-80 GB', bandwidth: '~1-2 TB/s', latency: '~200-800 cycles', color: '#f85149' },
  { name: 'L2 Cache', size: '~40 MB', bandwidth: '~3-5 TB/s', latency: '~200 cycles', color: '#ffab00' },
  { name: 'Shared Memory', size: '48-164 KB/SM', bandwidth: '~19 TB/s', latency: '~20-30 cycles', color: '#76B900' },
  { name: 'Registers', size: '255/thread', bandwidth: '~80 TB/s', latency: '~1 cycle', color: '#8ce600' }
]

const selectedLevel = ref(2)  // Default: Shared Memory

// Bank conflict visualization
const bankCount = 32
const paddingEnabled = ref(false)

const bankData = computed(() => {
  const banks = new Array(bankCount).fill(0)
  for (let i = 0; i < 32; i++) {
    const bankIdx = paddingEnabled.value
      ? (i * 5) % bankCount  // With padding, different mapping
      : (i * 4) % bankCount  // Without padding, some conflict
    banks[bankIdx]++
  }
  return banks
})

const hasConflicts = computed(() => bankData.value.some(b => b > 1))

const currentLevel = computed(() => levels[selectedLevel.value])
</script>

<template>
  <div class="kernel-memory-viz">
    <h4 class="section-title">GPU Memory Hierarchy Visualization</h4>

    <!-- Memory hierarchy pyramid -->
    <div class="hierarchy-pyramid">
      <div
        v-for="(level, idx) in levels"
        :key="level.name"
        class="hierarchy-level"
        :class="{ selected: selectedLevel === idx }"
        :style="{
          width: `${40 + idx * 20}%`,
          backgroundColor: level.color + '20',
          borderColor: selectedLevel === idx ? level.color : 'var(--vp-c-divider)'
        }"
        @click="selectedLevel = idx"
      >
        <span class="level-name">{{ level.name }}</span>
        <span class="level-size">{{ level.size }}</span>
      </div>
    </div>

    <!-- Selected level details -->
    <div class="level-details">
      <h5>{{ currentLevel.name }}</h5>
      <div class="detail-grid">
        <div class="detail-item">
          <span class="detail-label">Size</span>
          <span class="detail-value">{{ currentLevel.size }}</span>
        </div>
        <div class="detail-item">
          <span class="detail-label">Bandwidth</span>
          <span class="detail-value">{{ currentLevel.bandwidth }}</span>
        </div>
        <div class="detail-item">
          <span class="detail-label">Latency</span>
          <span class="detail-value">{{ currentLevel.latency }}</span>
        </div>
      </div>
    </div>

    <!-- Bank conflict visualization (for Shared Memory) -->
    <div v-if="selectedLevel === 2" class="bank-conflict-section">
      <h5>Shared Memory Bank Conflicts</h5>

      <div class="padding-toggle">
        <label>
          <input type="checkbox" v-model="paddingEnabled" />
          <span>Enable Padding (BK+1)</span>
        </label>
        <span class="conflict-status" :class="{ good: !hasConflicts, bad: hasConflicts }">
          {{ hasConflicts ? 'Bank Conflicts Detected!' : 'No Bank Conflicts' }}
        </span>
      </div>

      <div class="bank-grid">
        <div
          v-for="(count, idx) in bankData"
          :key="idx"
          class="bank-cell"
          :class="{ conflict: count > 1, empty: count === 0 }"
        >
          <span class="bank-index">{{ idx }}</span>
          <span class="bank-count">{{ count > 0 ? count + '×' : '-' }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.kernel-memory-viz {
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

.hierarchy-pyramid {
  display: flex;
  flex-direction: column-reverse;
  align-items: center;
  gap: 4px;
  margin-bottom: 24px;
}

.hierarchy-level {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 20px;
  border: 2px solid;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.hierarchy-level:hover {
  transform: scale(1.02);
}

.hierarchy-level.selected {
  transform: scale(1.05);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.level-name {
  font-weight: 600;
  font-size: 0.9rem;
}

.level-size {
  font-family: var(--vp-font-family-mono);
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
}

.level-details {
  padding: 16px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  margin-bottom: 20px;
}

.level-details h5 {
  margin: 0 0 12px 0;
  font-size: 1rem;
  color: var(--vp-c-brand-1);
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}

.detail-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.detail-label {
  font-size: 0.8rem;
  color: var(--vp-c-text-3);
}

.detail-value {
  font-family: var(--vp-font-family-mono);
  font-size: 0.9rem;
  color: var(--vp-c-text-1);
}

.bank-conflict-section {
  padding: 16px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
}

.bank-conflict-section h5 {
  margin: 0 0 12px 0;
  font-size: 0.95rem;
  color: var(--vp-c-text-1);
}

.padding-toggle {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}

.padding-toggle label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
}

.padding-toggle input[type="checkbox"] {
  accent-color: var(--vp-c-brand-1);
}

.conflict-status {
  font-size: 0.85rem;
  font-weight: 600;
}

.conflict-status.good {
  color: var(--vp-c-brand-1);
}

.conflict-status.bad {
  color: #f85149;
}

.bank-grid {
  display: grid;
  grid-template-columns: repeat(16, 1fr);
  gap: 4px;
}

.bank-cell {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 6px 2px;
  background: var(--vp-c-bg-alt);
  border-radius: 4px;
  border: 1px solid var(--vp-c-divider);
  font-size: 0.7rem;
}

.bank-cell.conflict {
  background: rgba(248, 81, 73, 0.2);
  border-color: #f85149;
}

.bank-cell.empty {
  opacity: 0.4;
}

.bank-index {
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
}

.bank-count {
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.bank-cell.conflict .bank-count {
  color: #f85149;
}
</style>
