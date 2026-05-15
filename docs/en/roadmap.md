---
outline: [2, 3]
---

# Roadmap

Use this roadmap to choose a learning track, not just to see a chronological list of topics.

- **If you want one fast win**, use the entry table and stop after the first stage that matches your goal.
- **If you want a full academy path**, follow the four stages in order.
- **If you want system intuition**, read the module, benchmark, and whitepaper links together instead of treating this page as standalone.

## Choose your entry point

| If your immediate goal is... | Start with | Why | Then read |
| --- | --- | --- | --- |
| Get comfortable reading CUDA kernels | [01-SGEMM Tutorial](/en/modules/01-sgemm) | It keeps the optimization story explicit and local. | [Benchmarks](/en/benchmarks/) |
| Build reusable CUDA abstractions | [02-TensorCraft Core](/en/modules/02-tensorcraft) | It moves from kernel tricks to reusable interfaces and runtime helpers. | [TensorCraft design](/en/whitepaper/tensorcraft-design) |
| Chase higher-performance techniques | [03-HPC Advanced](/en/modules/03-hpc) | It is where the repo starts to discuss more architecture-sensitive ideas. | [Advanced showcase](/en/whitepaper/advanced-showcase) |
| Understand end-to-end inference assembly | [04-Inference Engine](/en/modules/04-inference) | It connects optimized kernels to memory pools, streams, and runtime orchestration. | [Inference engine design](/en/whitepaper/inference-engine-design) |

## Stage map

```mermaid
journey
    title CUDA Kernel Academy Learning Path
    section Stage 1: Fundamentals
      SGEMM Basics: 5: Naive GEMM
      Shared Memory: 5: Tiling and Bank Conflict
      Double Buffer: 4: Hiding memory latency
    section Stage 2: Core Library
      TensorCraft Core: 5: Industrial API design
      Modern C++ CUDA: 4: Reusable interfaces
      Kernel Fusion: 4: Fusion and epilogue thinking
    section Stage 3: Advanced HPC
      GEMM Optimization: 5: Register tiling / WMMA
      FlashAttention: 4: IO-aware kernels
      CUDA 12+ Features: 3: New hardware-facing features
    section Stage 4: Inference System
      Inference Engine: 5: Memory pool / Stream / Tensor
      Performance Tuning: 4: AutoTuner / profiling
      Quantized Inference: 3: Deployment-oriented trade-offs
```

## What each stage is trying to teach

| Stage | Main question | Primary repo pages | Evidence to look for | Foundational paper or reference |
| --- | --- | --- | --- | --- |
| 1. Fundamentals | How does a fast kernel emerge from a simple one? | [01-SGEMM Tutorial](/en/modules/01-sgemm), [Benchmarks](/en/benchmarks/) | Stepwise gains in TFLOPS, bandwidth, and correctness confidence | Boehm matmul article; CUDA Programming Guide |
| 2. Core Library | Which optimization ideas deserve reusable abstractions? | [02-TensorCraft Core](/en/modules/02-tensorcraft), [Architecture](/en/whitepaper/architecture), [TensorCraft design](/en/whitepaper/tensorcraft-design) | Library boundaries that stay readable without losing the optimization story | CUDA C++ Programming Guide; CUTLASS mental model |
| 3. Advanced HPC | How do you move closer to hardware ceilings? | [03-HPC Advanced](/en/modules/03-hpc), [Advanced showcase](/en/whitepaper/advanced-showcase) | Larger gains from register tiling, WMMA, FlashAttention-like ideas, or newer CUDA features | FlashAttention paper; CUDA feature docs |
| 4. Inference System | How do optimized kernels survive contact with a real runtime? | [04-Inference Engine](/en/modules/04-inference), [Inference engine design](/en/whitepaper/inference-engine-design) | Throughput and orchestration evidence that the system still benefits from earlier kernel work | System design docs plus benchmark interpretation |

## How the modules connect over time

```mermaid
flowchart LR
    A[Learn one kernel deeply] --> B[Abstract reusable parts]
    B --> C[Push techniques toward hardware limits]
    B --> D[Integrate into runtime system]
    C --> D
```

The point of the roadmap is not merely sequential difficulty. It reflects a change in what “success” means:

- In **Stage 1**, success means understanding *why* each optimization step exists.
- In **Stage 2**, success means keeping those ideas reusable without losing clarity.
- In **Stage 3**, success means knowing when architecture-specific techniques are worth the complexity.
- In **Stage 4**, success means proving the earlier work survives system integration.

## Suggested study plans

### 3-day orientation pass

- Day 1: [Getting Started](/en/guides/getting-started) + [01-SGEMM Tutorial](/en/modules/01-sgemm)
- Day 2: [Benchmarks](/en/benchmarks/) + [System Architecture](/en/whitepaper/architecture)
- Day 3: Choose either [02-TensorCraft Core](/en/modules/02-tensorcraft) or [04-Inference Engine](/en/modules/04-inference) based on whether you care more about library design or runtime integration.

### 2-week kernel engineering track

- Week 1: Finish Stage 1 and capture the optimization ladder in your own notes.
- Week 2: Read Stage 2 plus the relevant whitepapers, then sample Stage 3 topics that match your GPU and interests.

### Full academy path

- Move through all four stages in order.
- After each stage, verify your understanding against the linked benchmark or whitepaper page before advancing.
- Do not skip directly from Stage 1 to Stage 4 unless your goal is purely system orientation; you will miss why the abstractions exist.

## Foundational references by stage

<ReferenceBlock
  :references="[
    {
      id: '1',
      authors: 'Boehm, Simon',
      title: 'How to Optimize a CUDA Matmul Kernel',
      venue: 'Technical article',
      year: 2022,
      url: 'https://siboehm.com/articles/22/CUDA-MMM'
    },
    {
      id: '2',
      authors: 'Volkov, V. and Demmel, J. W.',
      title: 'Benchmarking GPUs to Tune Dense Linear Algebra',
      venue: 'SC',
      year: 2008
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

## What to study next after this page

- Need the big-picture rationale? Open [System Architecture](/en/whitepaper/architecture).
- Need performance evidence? Pair this page with [Benchmarks](/en/benchmarks/).
- Need to act on a concrete goal? Jump to the module linked in the entry table above.
