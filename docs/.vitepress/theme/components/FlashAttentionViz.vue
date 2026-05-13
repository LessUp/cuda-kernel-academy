<script setup lang="ts">
const standardSteps = [
  { label: 'Compute S = Q·Kᵀ', read: '2N·d', write: 'N²' },
  { label: 'Softmax(P)', read: 'N²', write: 'N²' },
  { label: 'Compute O = P·V', read: '2N·d + N²', write: 'N·d' }
]

const flashSteps = [
  { label: 'Tiling into SRAM blocks', detail: 'Block size ≈ SRAM capacity' },
  { label: 'Online Softmax', detail: 'Fused in SRAM, no full materialization' },
  { label: 'Write only final O', detail: 'HBM writes ≈ O(N·d)' }
]
</script>

<template>
  <div class="flash-attention-viz">
    <div class="comparison-row">
      <!-- 左侧：标准 Attention -->
      <div class="panel standard-panel">
        <div class="panel-header">
          <h3 class="panel-title">标准 Attention</h3>
          <div class="hbm-count red">O(N²)</div>
        </div>
        <div class="panel-subtitle">HBM 访问次数（序列长度 N 的平方级）</div>

        <div class="steps">
          <div v-for="(step, i) in standardSteps" :key="i" class="step-box red-border">
            <div class="step-label">{{ step.label }}</div>
            <div class="step-io">
              <span class="io-tag read">读 {{ step.read }}</span>
              <span class="io-tag write">写 {{ step.write }}</span>
            </div>
          </div>
        </div>

        <div class="panel-footer red">
          瓶颈：N² 级 HBM 读写，内存带宽受限
        </div>
      </div>

      <!-- 中间对比箭头 -->
      <div class="vs-column">
        <div class="vs-badge">
          <span class="vs-text">VS</span>
        </div>
        <div class="vs-arrow">
          <div class="arrow-line"></div>
          <div class="arrow-head"></div>
        </div>
        <div class="reduction-label">
          <span class="reduction-text">HBM 访问量</span>
          <span class="reduction-value">↓ 90%+</span>
        </div>
      </div>

      <!-- 右侧：Flash Attention -->
      <div class="panel flash-panel">
        <div class="panel-header">
          <h3 class="panel-title">FlashAttention</h3>
          <div class="hbm-count green">O(N)</div>
        </div>
        <div class="panel-subtitle">HBM 访问次数（序列长度 N 的线性级）</div>

        <div class="steps">
          <div v-for="(step, i) in flashSteps" :key="i" class="step-box green-border">
            <div class="step-label">{{ step.label }}</div>
            <div class="step-detail">{{ step.detail }}</div>
          </div>
        </div>

        <div class="sram-band">
          <div class="sram-label">SRAM 计算流程</div>
          <div class="sram-blocks">
            <div class="sram-block">Q<sub>i</sub></div>
            <div class="sram-block">K<sub>j</sub></div>
            <div class="sram-block">V<sub>j</sub></div>
          </div>
        </div>

        <div class="panel-footer green">
          优势：融合内核，减少 HBM 往返，逼近 SRAM 带宽上限
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.flash-attention-viz {
  margin: 24px 0;
}

.comparison-row {
  display: flex;
  gap: 16px;
  align-items: stretch;
}

.panel {
  flex: 1;
  background: var(--vp-c-bg-alt, #161b22);
  border: 1px solid var(--vp-c-border, #30363d);
  border-radius: 12px;
  padding: 24px;
  display: flex;
  flex-direction: column;
}

.standard-panel {
  border-top: 3px solid #f85149;
}

.flash-panel {
  border-top: 3px solid #76B900;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}

.panel-title {
  font-size: 1.1rem;
  font-weight: 700;
  margin: 0;
  color: var(--vp-c-text-1, #e6edf3);
}

.hbm-count {
  font-size: 1.5rem;
  font-weight: 800;
  font-family: var(--vp-font-family-mono);
}

.hbm-count.red {
  color: #f85149;
}

.hbm-count.green {
  color: #76B900;
}

.panel-subtitle {
  font-size: 0.85rem;
  color: var(--vp-c-text-2, #8b949e);
  margin-bottom: 16px;
}

.steps {
  display: flex;
  flex-direction: column;
  gap: 12px;
  flex: 1;
}

.step-box {
  background: var(--vp-c-bg, #0d1117);
  border-radius: 8px;
  padding: 14px 16px;
  border-left: 3px solid #30363d;
}

.red-border {
  border-left-color: #f85149;
}

.green-border {
  border-left-color: #76B900;
}

.step-label {
  font-weight: 600;
  font-size: 0.9rem;
  color: var(--vp-c-text-1, #e6edf3);
  margin-bottom: 6px;
}

.step-io {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.io-tag {
  font-size: 0.75rem;
  padding: 3px 8px;
  border-radius: 4px;
  font-family: var(--vp-font-family-mono);
}

.io-tag.read {
  background: rgba(248, 81, 73, 0.12);
  color: #f85149;
}

.io-tag.write {
  background: rgba(255, 171, 0, 0.12);
  color: #ffab00;
}

.step-detail {
  font-size: 0.8rem;
  color: var(--vp-c-text-2, #8b949e);
  margin-top: 4px;
}

.sram-band {
  margin-top: 16px;
  background: rgba(118, 185, 0, 0.06);
  border: 1px dashed rgba(118, 185, 0, 0.3);
  border-radius: 8px;
  padding: 12px;
}

.sram-label {
  font-size: 0.75rem;
  font-weight: 600;
  color: #76B900;
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.sram-blocks {
  display: flex;
  gap: 8px;
}

.sram-block {
  flex: 1;
  background: rgba(118, 185, 0, 0.12);
  color: #76B900;
  text-align: center;
  padding: 8px;
  border-radius: 6px;
  font-size: 0.85rem;
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
}

.panel-footer {
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid var(--vp-c-border, #30363d);
  font-size: 0.85rem;
  font-weight: 500;
}

.panel-footer.red {
  color: #f85149;
}

.panel-footer.green {
  color: #76B900;
}

/* VS 中间列 */
.vs-column {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  width: 80px;
  flex-shrink: 0;
}

.vs-badge {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: var(--vp-c-bg-alt, #161b22);
  border: 1px solid var(--vp-c-border, #30363d);
  display: flex;
  align-items: center;
  justify-content: center;
}

.vs-text {
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--vp-c-text-2, #8b949e);
}

.vs-arrow {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1;
  min-height: 60px;
}

.arrow-line {
  width: 2px;
  flex: 1;
  background: linear-gradient(to bottom, #f85149, #76B900);
}

.arrow-head {
  width: 0;
  height: 0;
  border-left: 6px solid transparent;
  border-right: 6px solid transparent;
  border-top: 8px solid #76B900;
}

.reduction-label {
  text-align: center;
}

.reduction-text {
  display: block;
  font-size: 0.7rem;
  color: var(--vp-c-text-2, #8b949e);
  margin-bottom: 2px;
}

.reduction-value {
  display: block;
  font-size: 0.9rem;
  font-weight: 700;
  color: #76B900;
  font-family: var(--vp-font-family-mono);
}

@media (max-width: 768px) {
  .comparison-row {
    flex-direction: column;
  }
  .vs-column {
    flex-direction: row;
    width: 100%;
    padding: 8px 0;
  }
  .vs-arrow {
    flex-direction: row;
    min-height: auto;
    min-width: 40px;
  }
  .arrow-line {
    width: auto;
    height: 2px;
    flex: 1;
    background: linear-gradient(to right, #f85149, #76B900);
  }
  .arrow-head {
    border-top: 6px solid transparent;
    border-bottom: 6px solid transparent;
    border-left: 8px solid #76B900;
    border-right: none;
  }
}
</style>
