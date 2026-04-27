# Documentation Overview

This directory contains the small set of durable documents that explain how to build, use, and maintain CUDA Kernel Academy.

## Start here

| Goal | Document |
| --- | --- |
| understand the repository at a glance | [../README.md](../README.md) |
| build the repository correctly | [INSTALLATION.md](INSTALLATION.md) |
| understand the contribution and review flow | [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md) |
| understand AI-assisted workflow choices | [AI_TOOLING.md](AI_TOOLING.md) |
| follow the coding rules | [CODING_STYLE.md](CODING_STYLE.md) |
| debug common problems | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |

## Core repo docs

- [Development Workflow](DEVELOPMENT_WORKFLOW.md)
- [AI Tooling Guide](AI_TOOLING.md)
- [Installation Guide](INSTALLATION.md)
- [Coding Style](CODING_STYLE.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [Integration Examples](integration-examples.md)

## Module entry points

| Module | Entry point |
| --- | --- |
| 01 | [../01-sgemm-tutorial/README.md](../01-sgemm-tutorial/README.md) |
| 02 | [../02-tensorcraft-core/README.md](../02-tensorcraft-core/README.md) |
| 03 | [../03-hpc-advanced/README.md](../03-hpc-advanced/README.md) |
| 04 | [../04-inference-engine/README.md](../04-inference-engine/README.md) |

## Deep dives

- [TensorCraft Architecture](../02-tensorcraft-core/docs/architecture.md)
- [TensorCraft API Reference](../02-tensorcraft-core/docs/api_reference.md)
- [TensorCraft Optimization Guide](../02-tensorcraft-core/docs/optimization_guide.md)
- [HPC GEMM Optimization](../03-hpc-advanced/docs/01_gemm_optimization.md)
- [HPC Flash Attention](../03-hpc-advanced/docs/04_flash_attention.md)
- [Inference Engine Architecture](../04-inference-engine/docs/ARCHITECTURE.md)
- [Inference Engine Performance Tuning](../04-inference-engine/docs/PERFORMANCE_TUNING.md)

## Repository posture

The repository is intentionally biased toward **high-signal maintenance**:

- one OpenSpec workflow
- concise, repository-specific docs
- lean automation
- real GPU validation left to local machines
