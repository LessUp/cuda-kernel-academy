---
layout: home

hero:
  name: "CUDA Kernel Academy"
  text: "Systematic CUDA kernel engineering"
  tagline: "从 SGEMM 基础到可复用推理组件的仓库级学习地图，而不只是项目简介。"
  actions:
    - theme: brand
      text: "访问 GitHub"
      link: "https://github.com/AICL-Lab/cuda-kernel-academy"
    - theme: alt
      text: "开始阅读"
      link: "/zh/guides/getting-started"

features:
  - icon: <span class="feature-icon">01</span>
    title: SGEMM 教程
    details: 从 naive、tiling、bank conflict 修正、double buffering、向量化到 Tensor Core，完整展示算子优化阶梯。
    link: /zh/modules/01-sgemm
  - icon: <span class="feature-icon">02</span>
    title: TensorCraft Core
    details: 把教程里的 kernel 沉淀为可复用库接口，聚焦现代 C++ CUDA、RAII、错误处理和内存管理。
    link: /zh/modules/02-tensorcraft
  - icon: <span class="feature-icon">03</span>
    title: HPC 进阶
    details: 在仓库语境下继续推进到 CUTLASS、FlashAttention、寄存器分块和新 CUDA 特性。
    link: /zh/modules/03-hpc
  - icon: <span class="feature-icon">04</span>
    title: Inference Engine
    details: 观察优化后的 kernel 如何进入带内存池、流调度和端到端执行路径的推理系统。
    link: /zh/modules/04-inference
---

<div style="display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; max-width: 960px; margin: 32px auto;">
  <MetricBadge value="4" label="核心模块" />
  <MetricBadge value="3" label="阅读视角" />
  <MetricBadge value="2" label="构建系统" />
  <MetricBadge value="1" label="完整学习路径" />
</div>

## 这个首页是给谁看的

这不是一个“仓库简介页”，而是帮助你快速建立阅读顺序的入口页。

- **CUDA 新读者** 可以先确定学习顺序，而不是随机点开目录。
- **正在做性能优化的工程师** 可以直接找到最相关的模块和 benchmark 证据。
- **关心系统集成的读者** 可以快速看清教程 kernel、可复用基础设施和推理引擎之间的关系。

## 把这个仓库当成 academy 来读，而不是 brochure

建议把仓库分成三个互相补充的视角：

1. **模块文档**：告诉你每个目录在教什么、先看哪些代码。
2. **白皮书**：解释为什么仓库要这样拆分，以及模块之间怎样衔接。
3. **Benchmarks 与路线图**：说明性能证据意味着什么，以及下一步该读什么。

::: tip 推荐的第一遍阅读顺序
如果你是第一次接触这个仓库，建议先看 [快速开始](/zh/guides/getting-started)，再读 [01-SGEMM 教程](/zh/modules/01-sgemm)，之后回到更高层的 [系统架构白皮书](/zh/whitepaper/architecture)。
:::

## 按目标选择第一站

| 你的目标 | 建议先读 | 为什么从这里开始 | 下一步 |
| --- | --- | --- | --- |
| 理解 CUDA 性能优化基础 | [01-SGEMM 教程](/zh/modules/01-sgemm) | 它用最小但完整的教学面展示 kernel 优化阶梯。 | [Benchmarks](/zh/benchmarks/) → [SGEMM 白皮书](/zh/whitepaper/sgemm-optimization) |
| 设计可复用 CUDA 算子库 | [02-TensorCraft Core](/zh/modules/02-tensorcraft) | 它把“单个 kernel”提升为接口、资源管理和库边界设计。 | [系统架构](/zh/whitepaper/architecture) → [TensorCraft 设计](/zh/whitepaper/tensorcraft-design) |
| 学习更高级的 kernel 技术 | [03-HPC 进阶](/zh/modules/03-hpc) | 这里才开始讨论 CUTLASS、FlashAttention、寄存器分块和新 CUDA 特性。 | [高级技术展示](/zh/whitepaper/advanced-showcase) → [Benchmarks](/zh/benchmarks/) |
| 理解端到端推理系统形态 | [04-Inference Engine](/zh/modules/04-inference) | 它说明优化后的 kernel 如何进入 stream、memory pool 和推理执行链路。 | [推理引擎设计](/zh/whitepaper/inference-engine-design) → [路线图](/zh/roadmap) |

## 模块之间如何技术衔接

<SystemArchitectureDiagram />

这个仓库的层次是刻意设计的。

- **模块 01** 用最直接的方式讲清楚一个高性能 kernel 的基本机制。
- **模块 02** 提炼出教程里真正可复用、可维护的工程模式。
- **模块 03** 把关注点从“能跑且正确”推进到“逼近架构上限”。
- **模块 04** 则回答一个更系统的问题：前面这些优化如果不能接入完整推理流程，还有多少工程价值？

## 文档不同区域分别解决什么问题

| 文档区域 | 最适合什么时候看 | 它回答的核心问题 |
| --- | --- | --- |
| [模块文档](/zh/modules/01-sgemm) | 需要贴近代码阅读时 | “这个目录教什么？我应该先打开哪些代码？” |
| [白皮书](/zh/whitepaper/architecture) | 需要跨模块理解时 | “仓库为什么这样组织？模块之间如何互相供给？” |
| [Benchmarks](/zh/benchmarks/) | 需要判断证据强弱时 | “这些性能结果到底说明了什么，又没说明什么？” |
| [路线图](/zh/roadmap) | 需要安排学习顺序时 | “按我当前目标，下一步应该看哪里？” |

## 第一轮阅读最重要的外部参考

<ReferenceBlock
  :references="[
    {
      id: '1',
      authors: 'Volkov, V. and Demmel, J. W.',
      title: 'Benchmarking GPUs to Tune Dense Linear Algebra',
      venue: 'SC',
      year: 2008
    },
    {
      id: '2',
      authors: 'Boehm, Simon',
      title: 'How to Optimize a CUDA Matmul Kernel',
      venue: 'Technical article',
      year: 2022,
      url: 'https://siboehm.com/articles/22/CUDA-MMM'
    },
    {
      id: '3',
      authors: 'Dao, Tri et al.',
      title: 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness',
      venue: 'NeurIPS',
      year: 2022,
      url: 'https://arxiv.org/abs/2205.14135'
    },
    {
      id: '4',
      authors: 'NVIDIA',
      title: 'CUDA C++ Programming Guide',
      venue: 'CUDA Toolkit documentation',
      year: 2024,
      url: 'https://docs.nvidia.com/cuda/cuda-c-programming-guide/'
    }
  ]"
/>

## 接下来读什么

- 想先建立全局视图：继续读 [系统架构白皮书](/zh/whitepaper/architecture)。
- 想先看证据：打开 [Benchmarks](/zh/benchmarks/)，并把 01-SGEMM 模块页面一起对照。
- 想按步骤推进：跟着 [路线图](/zh/roadmap) 选择适合你的学习轨道。
