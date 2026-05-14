---
layout: home
---

<script setup>
import { onMounted, ref } from 'vue'

const redirecting = ref(false)

onMounted(() => {
  // Auto-detect language preference
  const lang = navigator.language.toLowerCase()
  const preferZh = lang.startsWith('zh')

  if (preferZh) {
    redirecting.value = true
    setTimeout(() => {
      window.location.href = '/cuda-kernel-academy/zh/'
    }, 300)
  } else {
    redirecting.value = true
    setTimeout(() => {
      window.location.href = '/cuda-kernel-academy/en/'
    }, 300)
  }
})
</script>

<div v-if="redirecting" class="language-redirect">
  <div class="redirect-content">
    <div class="spinner"></div>
    <p>Redirecting to your preferred language...</p>
  </div>
</div>

<div v-else class="language-selector">
  <div class="selector-content">
    <h1>CUDA Kernel Academy</h1>
    <p class="tagline">Systematic CUDA kernel engineering from SGEMM to inference</p>
    <p class="tagline-zh">从 SGEMM 基础到可复用推理组件的系统性 CUDA 算子工程学习路径</p>

    <div class="language-buttons">
      <a href="/cuda-kernel-academy/zh/" class="lang-btn">
        <span class="lang-icon">中</span>
        <span class="lang-label">简体中文</span>
      </a>
      <a href="/cuda-kernel-academy/en/" class="lang-btn">
        <span class="lang-icon">EN</span>
        <span class="lang-label">English</span>
      </a>
    </div>
  </div>
</div>

<style scoped>
.language-redirect,
.language-selector {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 60vh;
  text-align: center;
}

.redirect-content,
.selector-content {
  max-width: 600px;
  padding: 40px;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--vp-c-divider);
  border-top-color: #76B900;
  border-radius: 50%;
  margin: 0 auto 20px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

h1 {
  font-size: 2.5rem;
  font-weight: 800;
  margin-bottom: 16px;
  background: linear-gradient(135deg, #76B900, #8ce600);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.tagline {
  font-size: 1.1rem;
  color: var(--vp-c-text-2);
  margin-bottom: 8px;
}

.tagline-zh {
  font-size: 1rem;
  color: var(--vp-c-text-3);
  margin-bottom: 40px;
}

.language-buttons {
  display: flex;
  gap: 20px;
  justify-content: center;
  flex-wrap: wrap;
}

.lang-btn {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 24px 48px;
  border: 2px solid var(--vp-c-divider);
  border-radius: 12px;
  text-decoration: none;
  transition: all 0.3s ease;
  background: var(--vp-c-bg-alt);
}

.lang-btn:hover {
  border-color: #76B900;
  transform: translateY(-4px);
  box-shadow: 0 12px 40px rgba(118, 185, 0, 0.15);
}

.lang-icon {
  font-size: 2rem;
  font-weight: 700;
  color: #76B900;
  margin-bottom: 8px;
}

.lang-label {
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
}
</style>
