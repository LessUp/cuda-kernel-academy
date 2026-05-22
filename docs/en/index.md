---
layout: home

hero:
  name: "CUDA Kernel Academy"
  text: "Systematic CUDA kernel engineering"
  tagline: "A repository-specific reading map from SGEMM fundamentals to reusable inference components."
  actions:
    - theme: brand
      text: "Visit GitHub"
      link: "https://github.com/AICL-Lab/cuda-kernel-academy"
    - theme: alt
      text: "Start Reading"
      link: "/en/guides/getting-started"

features:
  - icon: <span class="feature-icon">01</span>
    title: SGEMM Tutorial
    details: "Learn the kernel optimization ladder: naive, tiling, bank-conflict cleanup, double buffering, vectorization, and Tensor Core variants."
    link: /en/modules/01-sgemm
  - icon: <span class="feature-icon">02</span>
    title: TensorCraft Core
    details: "See how tutorial kernels become reusable library components with modern C++ CUDA interfaces, error handling, and memory management."
    link: /en/modules/02-tensorcraft
  - icon: <span class="feature-icon">03</span>
    title: HPC Advanced
    details: "Study the module that pushes toward production-grade performance with CUTLASS, FlashAttention, register tiling, and newer CUDA features."
    link: /en/modules/03-hpc
  - icon: <span class="feature-icon">04</span>
    title: Inference Engine
    details: "Follow the end-to-end system path where optimized kernels become a schedulable inference pipeline with memory pools and streams."
    link: /en/modules/04-inference
---

<div style="display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; max-width: 960px; margin: 32px auto;">
  <MetricBadge value="4" label="Core modules" />
  <MetricBadge value="3" label="Reading lenses" />
  <MetricBadge value="2" label="Build systems" />
  <MetricBadge value="1" label="Integrated learning path" />
</div>

## Who this landing page is for

This page is the fastest orientation pass for readers who want more than a repo brochure.

- **New CUDA learners** can use it as a staged reading order instead of opening modules at random.
- **Practitioners tuning kernels** can jump straight to the module and benchmark evidence most relevant to GEMM performance.
- **Systems builders** can see how tutorial kernels, reusable infrastructure, and inference integration relate inside one repository.

## Read the repository like an academy, not a brochure

Treat the repository as three connected views of the same work:

1. **Module pages** explain what each directory teaches and what code to inspect first.
2. **Whitepapers** explain why the modules are shaped the way they are and how the pieces connect.
3. **Benchmarks and roadmap** tell you how to validate the results and what to study next.

::: tip Recommended first pass
If you are new to this repo, read [Getting Started](/en/guides/getting-started), then skim [01-SGEMM Tutorial](/en/modules/01-sgemm), and only then come back to the broader [architecture whitepaper](/en/whitepaper/architecture).
:::

## Choose your first module by goal

| Your goal | Start here | Why this is the right entry | Read next |
| --- | --- | --- | --- |
| Understand CUDA performance basics | [01-SGEMM Tutorial](/en/modules/01-sgemm) | It shows the optimization ladder in the smallest possible teaching surface. | [Benchmarks](/en/benchmarks/) → [SGEMM whitepaper](/en/whitepaper/sgemm-optimization) |
| Build reusable CUDA operators | [02-TensorCraft Core](/en/modules/02-tensorcraft) | It turns isolated kernels into reusable abstractions, RAII-based resource handling, and library boundaries. | [Architecture](/en/whitepaper/architecture) → [TensorCraft design](/en/whitepaper/tensorcraft-design) |
| Study advanced kernel techniques | [03-HPC Advanced](/en/modules/03-hpc) | It is where register tiling, FlashAttention, CUTLASS, and newer CUDA features are discussed in repo context. | [Advanced showcase](/en/whitepaper/advanced-showcase) → [Benchmarks](/en/benchmarks/) |
| Understand end-to-end deployment shape | [04-Inference Engine](/en/modules/04-inference) | It shows how optimized kernels feed stream scheduling, memory pools, and inference-oriented execution. | [Inference engine design](/en/whitepaper/inference-engine-design) → [Roadmap](/en/roadmap) |

## How the modules connect technically

<SystemArchitectureDiagram />

The repo is intentionally layered.

- **Module 01** teaches the mechanics of a fast kernel in the most explicit form.
- **Module 02** captures the reusable patterns that survive beyond a one-off tutorial kernel.
- **Module 03** explores techniques that matter when you are chasing architectural limits rather than just correctness.
- **Module 04** proves that the earlier modules are useful only if they can be integrated into a whole inference pipeline.

## What each documentation area gives you

| Documentation area | Best used for | Main question it answers |
| --- | --- | --- |
| [Module docs](/en/modules/01-sgemm) | Close reading of code and concepts | “What does this directory teach, and where should I open the code first?” |
| [Whitepapers](/en/whitepaper/architecture) | Cross-module reasoning | “Why is the repo structured this way, and how do the modules feed one another?” |
| [Benchmarks](/en/benchmarks/) | Evidence and limits | “What performance story is the repo claiming, and how should I interpret it?” |
| [Roadmap](/en/roadmap) | Sequencing your study | “What should I read next for my current goal and experience level?” |

## Foundational references for the first pass

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

## Study next

- Want the system view first? Read the [architecture whitepaper](/en/whitepaper/architecture).
- Want evidence before theory? Read [benchmarks](/en/benchmarks/) with the SGEMM module open beside it.
- Want a guided route by goal? Open [Reading Paths](/en/guides/reading-paths).
- Want a guided sequence? Follow the [roadmap](/en/roadmap) and choose the track that matches your goal.
