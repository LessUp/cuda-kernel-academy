<script setup lang="ts">
interface TimelineNode {
  version: string
  arch: string
  features: string[]
  highlights: number[] // indices of features to highlight
}

const nodes: TimelineNode[] = [
  {
    version: 'CUDA 9',
    arch: 'Volta',
    features: ['Tensor Core V1', 'CUDA Graphs', '协作组'],
    highlights: [0, 2]
  },
  {
    version: 'CUDA 10',
    arch: 'Turing',
    features: ['MIG 支持', '图优化 API', 'RT Core'],
    highlights: [1]
  },
  {
    version: 'CUDA 11',
    arch: 'Ampere',
    features: ['TF32 / BF16', '稀疏 Tensor Core', 'MPS 增强'],
    highlights: [0, 1]
  },
  {
    version: 'CUDA 12',
    arch: 'Hopper / Ada',
    features: ['DPX 指令', 'Transformer Engine', '线程块集群', '动态编程加速'],
    highlights: [1, 2, 3]
  },
  {
    version: 'CUDA 13',
    arch: 'Blackwell',
    features: ['FP4 / FP6', '第二代 Transformer Engine', '微张量缩放', 'AI 计算扩展'],
    highlights: [0, 1, 2, 3]
  }
]
</script>

<template>
  <div class="cuda-timeline">
    <div class="timeline-track">
      <div
        v-for="(node, index) in nodes"
        :key="node.version"
        class="timeline-node"
      >
        <div class="node-connector" v-if="index > 0">
          <div class="connector-line"></div>
        </div>
        <div class="node-card">
          <div class="node-version">{{ node.version }}</div>
          <div class="node-arch">{{ node.arch }}</div>
          <div class="node-features">
            <span
              v-for="(feat, fIndex) in node.features"
              :key="fIndex"
              class="feature-tag"
              :class="{ highlighted: node.highlights.includes(fIndex) }"
            >
              {{ feat }}
            </span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cuda-timeline {
  margin: 24px 0;
  overflow-x: auto;
  padding-bottom: 8px;
}

.timeline-track {
  display: flex;
  align-items: flex-start;
  gap: 0;
  min-width: 700px;
}

.timeline-node {
  display: flex;
  align-items: center;
  flex: 1;
}

.node-connector {
  display: flex;
  align-items: center;
  flex: 1;
  padding: 0 8px;
}

.connector-line {
  height: 3px;
  width: 100%;
  border-radius: 2px;
  background: linear-gradient(90deg, #5c9100, #76B900, #a8ff00);
  position: relative;
}

.connector-line::after {
  content: '';
  position: absolute;
  right: -6px;
  top: 50%;
  transform: translateY(-50%);
  width: 0;
  height: 0;
  border-top: 5px solid transparent;
  border-bottom: 5px solid transparent;
  border-left: 7px solid #a8ff00;
}

.node-card {
  background: var(--vp-c-bg-alt, #161b22);
  border: 1px solid var(--vp-c-border, #30363d);
  border-radius: 12px;
  padding: 18px;
  min-width: 140px;
  text-align: center;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
  cursor: default;
  flex-shrink: 0;
}

.node-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
  border-color: rgba(118, 185, 0, 0.5);
}

.node-version {
  font-size: 1.1rem;
  font-weight: 700;
  color: #76B900;
  margin-bottom: 4px;
  font-family: var(--vp-font-family-mono);
}

.node-arch {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--vp-c-text-2, #8b949e);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 10px;
}

.node-features {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.feature-tag {
  font-size: 0.75rem;
  padding: 4px 8px;
  border-radius: 4px;
  background: var(--vp-c-bg, #0d1117);
  color: var(--vp-c-text-2, #8b949e);
  border: 1px solid var(--vp-c-border, #30363d);
  transition: all 0.2s ease;
}

.feature-tag.highlighted {
  background: rgba(118, 185, 0, 0.12);
  color: #76B900;
  border-color: rgba(118, 185, 0, 0.4);
  font-weight: 600;
}

.node-card:hover .feature-tag.highlighted {
  background: rgba(118, 185, 0, 0.2);
}

@media (max-width: 768px) {
  .timeline-track {
    flex-direction: column;
    min-width: auto;
  }
  .timeline-node {
    width: 100%;
    flex-direction: column;
    align-items: stretch;
  }
  .node-connector {
    padding: 8px 0;
    justify-content: center;
  }
  .connector-line {
    width: 3px;
    height: 24px;
    background: linear-gradient(180deg, #5c9100, #76B900, #a8ff00);
  }
  .connector-line::after {
    right: auto;
    top: auto;
    bottom: -6px;
    left: 50%;
    transform: translateX(-50%);
    border-top: 7px solid #a8ff00;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-bottom: none;
  }
  .node-card {
    min-width: auto;
    width: 100%;
  }
}
</style>
