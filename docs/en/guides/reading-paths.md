---
outline: [2, 3]
---

# Reading Paths

Use this page when you do not want to read the repository in directory order.

## Path 1: First contact

Best for readers who want a fast orientation before opening code.

1. [Getting Started](./getting-started.md)
2. [01-SGEMM Tutorial](/en/modules/01-sgemm)
3. [System Architecture](/en/whitepaper/architecture)
4. [Roadmap](/en/roadmap)

## Path 2: Kernel optimization

Best for readers chasing performance mechanics first.

1. [01-SGEMM Tutorial](/en/modules/01-sgemm)
2. [SGEMM Optimization](/en/whitepaper/sgemm-optimization)
3. [03-HPC Advanced](/en/modules/03-hpc)
4. [Benchmarks](/en/benchmarks/)

## Path 3: Reusable CUDA library design

Best for readers more interested in interfaces, library shape, and reuse.

1. [02-TensorCraft Core](/en/modules/02-tensorcraft)
2. [TensorCraft Design](/en/whitepaper/tensorcraft-design)
3. [Architecture](/en/modules/02/architecture)
4. [API Reference](/en/modules/02/api-reference)

## Path 4: Inference system integration

Best for readers who care about how kernels become part of a runnable system.

1. [04-Inference Engine](/en/modules/04-inference)
2. [Inference Engine Design](/en/whitepaper/inference-engine-design)
3. [Quick Start](/en/modules/04/quick-start)
4. [Performance Tuning](/en/modules/04/performance-tuning)

## Verification reality

- Docs and Pages builds are CPU-safe.
- CUDA build and runtime validation still belong on a real GPU machine.
- Use [Workflow](./workflow.md) for the repo change loop and verification commands.
