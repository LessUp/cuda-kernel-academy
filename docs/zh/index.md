---
layout: home

hero:
  name: "CUDA Kernel Academy"
  text: "Systematic CUDA kernel engineering"
  tagline: "从 SGEMM 基础到可复用推理组件的系统性 CUDA 算子工程学习路径"
  actions:
    - theme: brand
      text: "访问 GitHub"
      link: "https://github.com/LessUp/cuda-kernel-academy"
    - theme: alt
      text: "开始阅读"
      link: "/zh/getting-started"

features:
  - icon: <span class="feature-icon">01</span>
    title: SGEMM 教程
    details: 从 naive 到分块、向量化、双缓冲的矩阵乘法渐进式优化，建立 CUDA 性能调优直觉。
    link: /zh/01-sgemm/
  - icon: <span class="feature-icon">02</span>
    title: TensorCraft Core
    details: 现代 C++ CUDA 算子库：表达式模板、自动求导、内存池与多流调度基础设施。
    link: /zh/02-tensorcraft/
  - icon: <span class="feature-icon">03</span>
    title: HPC 进阶
    details: CUTLASS 集成、Flash Attention、Warp 特化、CUDA 新特性与极致内存优化。
    link: /zh/03-hpc/
  - icon: <span class="feature-icon">04</span>
    title: Inference Engine
    details: 端到端推理引擎：GEMM 调优、量化融合、多批次调度与性能基准测试。
    link: /zh/04-inference/
---

<HeroSection
  tagline="从 SGEMM 基础到可复用推理组件的系统性 CUDA 算子工程学习路径"
  docs-link="/cuda-kernel-academy/zh/guides/getting-started"
  :labels="{ modules: '核心模块', tips: '优化技巧', peak: 'cuBLAS 峰值', engine: '推理引擎' }"
/>

<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; max-width: 900px; margin: 32px auto;">
  <MetricBadge value="4" label="核心模块" />
  <MetricBadge value="7" label="GEMM 变体" />
  <MetricBadge value="12+" label="优化技术" />
  <MetricBadge value="1" label="推理管线" />
</div>
