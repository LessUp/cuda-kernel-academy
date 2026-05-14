<script setup lang="ts">
import { ref, computed, watch } from 'vue'

// Tile size parameters
const BM = ref(128)
const BN = ref(128)
const BK = ref(8)
const TM = ref(8)
const TN = ref(8)

// Constraints
const maxThreadsPerBlock = 1024
const maxSharedMemBytes = 48 * 1024  // 48KB
const maxRegistersPerThread = 255

// Calculated values
const threadsPerBlock = computed(() => (BM.value / TM.value) * (BN.value / TN.value))
const sharedMemBytes = computed(() => (BM.value * BK.value + BK.value * BN.value) * 4)
const registersPerThread = computed(() => TM.value * TN.value + TM.value + TN.value + 10)

// Constraint checks
const threadsValid = computed(() => threadsPerBlock.value <= maxThreadsPerBlock)
const sharedMemValid = computed(() => sharedMemBytes.value <= maxSharedMemBytes)
const registersValid = computed(() => registersPerThread.value <= maxRegistersPerThread)
const allValid = computed(() => threadsValid.value && sharedMemValid.value && registersValid.value)

// Estimated performance (simplified model)
const estimatedGflops = computed(() => {
  if (!allValid.value) return 0
  // Base performance estimate based on tile sizes
  const efficiency = Math.min(1, (threadsPerBlock.value / 256)) * 0.8
  const compute = 2 * BM.value * BN.value * BK.value  // FLOPs per tile
  return Math.round(compute * efficiency * 0.001)  // Simplified GFLOPS estimate
})

// Preset configurations
const presets = [
  { name: 'Small', BM: 64, BN: 64, BK: 8, TM: 4, TN: 4 },
  { name: 'Medium', BM: 128, BN: 128, BK: 8, TM: 8, TN: 8 },
  { name: 'Large', BM: 128, BN: 128, BK: 16, TM: 8, TN: 8 },
  { name: 'Tensor Core', BM: 64, BN: 64, BK: 16, TM: 8, TN: 8 }
]

const applyPreset = (preset: typeof presets[0]) => {
  BM.value = preset.BM
  BN.value = preset.BN
  BK.value = preset.BK
  TM.value = preset.TM
  TN.value = preset.TN
}
</script>

<template>
  <div class="gemm-tiling-interactive">
    <h4 class="section-title">GEMM Tiling Parameter Explorer</h4>

    <!-- Presets -->
    <div class="presets">
      <button
        v-for="preset in presets"
        :key="preset.name"
        class="preset-btn"
        @click="applyPreset(preset)"
      >
        {{ preset.name }}
      </button>
    </div>

    <!-- Parameter sliders -->
    <div class="parameters">
      <div class="param-group">
        <label>
          <span class="param-name">BM (Block M)</span>
          <input type="range" v-model.number="BM" min="16" max="256" step="16" />
          <span class="param-value">{{ BM }}</span>
        </label>

        <label>
          <span class="param-name">BN (Block N)</span>
          <input type="range" v-model.number="BN" min="16" max="256" step="16" />
          <span class="param-value">{{ BN }}</span>
        </label>

        <label>
          <span class="param-name">BK (Block K)</span>
          <input type="range" v-model.number="BK" min="4" max="32" step="4" />
          <span class="param-value">{{ BK }}</span>
        </label>
      </div>

      <div class="param-group">
        <label>
          <span class="param-name">TM (Thread M)</span>
          <input type="range" v-model.number="TM" min="1" max="16" step="1" />
          <span class="param-value">{{ TM }}</span>
        </label>

        <label>
          <span class="param-name">TN (Thread N)</span>
          <input type="range" v-model.number="TN" min="1" max="16" step="1" />
          <span class="param-value">{{ TN }}</span>
        </label>
      </div>
    </div>

    <!-- Results -->
    <div class="results">
      <div class="result-item" :class="{ valid: threadsValid, invalid: !threadsValid }">
        <span class="result-label">Threads/Block</span>
        <span class="result-value">{{ threadsPerBlock }} / {{ maxThreadsPerBlock }}</span>
        <span class="result-status">{{ threadsValid ? '✓' : '✗' }}</span>
      </div>

      <div class="result-item" :class="{ valid: sharedMemValid, invalid: !sharedMemValid }">
        <span class="result-label">Shared Memory</span>
        <span class="result-value">{{ (sharedMemBytes / 1024).toFixed(1) }} KB / 48 KB</span>
        <span class="result-status">{{ sharedMemValid ? '✓' : '✗' }}</span>
      </div>

      <div class="result-item" :class="{ valid: registersValid, invalid: !registersValid }">
        <span class="result-label">Registers/Thread</span>
        <span class="result-value">{{ registersPerThread }} / {{ maxRegistersPerThread }}</span>
        <span class="result-status">{{ registersValid ? '✓' : '✗' }}</span>
      </div>
    </div>

    <!-- Performance estimate -->
    <div class="performance" :class="{ valid: allValid, invalid: !allValid }">
      <span class="perf-label">Configuration Status</span>
      <span class="perf-value">
        {{ allValid ? '✓ Valid Configuration' : '✗ Constraint Violation' }}
      </span>
    </div>

    <!-- Constraints reference -->
    <div class="constraints-info">
      <h5>Constraints</h5>
      <code>
        threads = (BM/TM) × (BN/TN) ≤ {{ maxThreadsPerBlock }}<br/>
        shared_mem = (BM×BK + BK×BN) × 4 ≤ {{ maxSharedMemBytes / 1024 }}KB<br/>
        registers = TM×TN + TM + TN + overhead ≤ {{ maxRegistersPerThread }}
      </code>
    </div>
  </div>
</template>

<style scoped>
.gemm-tiling-interactive {
  margin: 24px 0;
  padding: 24px;
  background: var(--vp-c-bg-alt);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
}

.section-title {
  margin: 0 0 16px 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.presets {
  display: flex;
  gap: 8px;
  margin-bottom: 20px;
}

.preset-btn {
  padding: 6px 16px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
  transition: all 0.2s;
}

.preset-btn:hover {
  border-color: #76B900;
  color: #76B900;
}

.parameters {
  display: grid;
  gap: 16px;
  margin-bottom: 20px;
}

.param-group {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
}

label {
  display: flex;
  align-items: center;
  gap: 8px;
}

.param-name {
  min-width: 100px;
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
}

input[type="range"] {
  flex: 1;
  accent-color: #76B900;
}

.param-value {
  min-width: 40px;
  font-family: var(--vp-font-family-mono);
  font-size: 0.85rem;
  color: #76B900;
}

.results {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}

.result-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px;
  background: var(--vp-c-bg);
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
}

.result-item.valid {
  border-color: #76B900;
}

.result-item.invalid {
  border-color: #f85149;
}

.result-label {
  flex: 1;
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
}

.result-value {
  font-family: var(--vp-font-family-mono);
  font-size: 0.85rem;
  color: var(--vp-c-text-1);
}

.result-status {
  font-weight: bold;
}

.valid .result-status {
  color: #76B900;
}

.invalid .result-status {
  color: #f85149;
}

.performance {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 16px;
  border-radius: 8px;
  font-weight: 600;
}

.performance.valid {
  background: rgba(118, 185, 0, 0.1);
  border: 1px solid #76B900;
  color: #76B900;
}

.performance.invalid {
  background: rgba(248, 81, 73, 0.1);
  border: 1px solid #f85149;
  color: #f85149;
}

.constraints-info {
  margin-top: 16px;
  padding: 12px;
  background: var(--vp-c-bg);
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
}

.constraints-info h5 {
  margin: 0 0 8px 0;
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
}

.constraints-info code {
  font-size: 0.8rem;
  color: var(--vp-c-text-3);
  line-height: 1.6;
}
</style>
