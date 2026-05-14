<template>
  <div class="hero-section">
    <div class="hero-content">
      <div class="hero-left">
        <h1 class="hero-title">
          <span class="gradient-text">CUDA Kernel</span><br />
          <span class="gradient-text">Academy</span>
        </h1>
        <p class="hero-tagline">
          {{ tagline }}
        </p>
        <div class="hero-actions">
          <a
            href="https://github.com/LessUp/cuda-kernel-academy"
            class="hero-btn hero-btn-primary"
            target="_blank"
            rel="noopener"
          >Visit GitHub</a>
          <a :href="docsLink" class="hero-btn hero-btn-secondary">Read Docs →</a>
        </div>
      </div>
      <div class="hero-right">
        <div class="code-preview">
          <div class="code-header">
            <span class="code-dot red"></span>
            <span class="code-dot yellow"></span>
            <span class="code-dot green"></span>
            <span class="code-filename">sgemm_kernel.cu</span>
          </div>
          <pre class="code-body"><code>__global__ void sgemm_v4(
  float* C, const float* A,
  const float* B, int M, int N, int K
) {
  // Tile with double buffering
  __shared__ float sA[2][BK * BM];
  __shared__ float sB[2][BK * BN];

  float rC[TM][TN] = {0};
  int stage = 0;

  #pragma unroll
  for (int t = 0; t &lt; K; t += BK) {
    // Async copy + compute overlap
    cp_async(&amp;sA[stage][...], &amp;A[...]);
    cp_async(&amp;sB[stage][...], &amp;B[...]);

    __syncthreads();
    stage ^= 1;

    #pragma unroll
    for (int i = 0; i &lt; TM; ++i)
      for (int j = 0; j &lt; TN; ++j)
        rC[i][j] += ...;
  }
  // Write back with vectorized store
  store_matrix_sync(C, rC);
}</code></pre>
        </div>
      </div>
    </div>

    <div class="hero-metrics">
      <div class="metric-badge">
        <div class="metric-value">4</div>
        <div class="metric-label">{{ labels.modules }}</div>
      </div>
      <div class="metric-badge">
        <div class="metric-value">40+</div>
        <div class="metric-label">{{ labels.tips }}</div>
      </div>
      <div class="metric-badge">
        <div class="metric-value">98%</div>
        <div class="metric-label">{{ labels.peak }}</div>
      </div>
      <div class="metric-badge">
        <div class="metric-value">0→1</div>
        <div class="metric-label">{{ labels.engine }}</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  tagline: string
  docsLink: string
  labels: {
    modules: string
    tips: string
    peak: string
    engine: string
  }
}

defineProps<Props>()
</script>

<style scoped>
.hero-content {
  position: relative;
  z-index: 1;
  max-width: 1200px;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 48px;
  flex-wrap: wrap;
}

.hero-left {
  flex: 1 1 400px;
}

.hero-title {
  font-size: 3.5rem;
  font-weight: 800;
  line-height: 1.1;
  margin: 0 0 20px 0;
  letter-spacing: -0.02em;
}

.gradient-text {
  background: linear-gradient(135deg, var(--vp-c-brand-1) 0%, var(--vp-c-brand-2) 40%, var(--vp-c-brand-3) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-tagline {
  font-size: 1.15rem;
  line-height: 1.7;
  color: var(--vp-c-text-2);
  margin: 0 0 32px 0;
}

.hero-actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.hero-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.95rem;
  text-decoration: none;
  transition: all 0.2s ease;
}

.hero-btn-primary {
  background: var(--vp-c-brand-1);
  color: var(--vp-c-bg);
  border: 1px solid var(--vp-c-brand-1);
}

.hero-btn-primary:hover {
  background: var(--vp-c-brand-2);
  border-color: var(--vp-c-brand-2);
}

.hero-btn-secondary {
  background: transparent;
  color: var(--vp-c-text-1);
  border: 1px solid var(--vp-c-border);
}

.hero-btn-secondary:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.hero-right {
  flex: 1 1 480px;
  max-width: 560px;
}

.code-preview {
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 24px 64px rgba(0, 0, 0, 0.5);
}

.code-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  background: #161b22;
  border-bottom: 1px solid #30363d;
}

.code-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}
.code-dot.red { background: #ff5f56; }
.code-dot.yellow { background: #ffbd2e; }
.code-dot.green { background: #27c93f; }

.code-filename {
  margin-left: 8px;
  font-size: 0.8rem;
  color: #8b949e;
  font-family: var(--vp-font-family-mono);
}

.code-body {
  margin: 0;
  padding: 16px;
  overflow-x: auto;
  font-size: 0.8rem;
  line-height: 1.6;
}

.code-body code {
  color: #e6edf3;
  font-family: var(--vp-font-family-mono);
}

.hero-metrics {
  position: relative;
  z-index: 1;
  display: flex;
  justify-content: center;
  gap: 48px;
  flex-wrap: wrap;
  margin-top: 48px;
  width: 100%;
  max-width: 900px;
}
</style>
