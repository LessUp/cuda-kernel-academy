---
outline: [2, 3]
---

# 03-HPC Advanced

Advanced CUDA / HPC tutorial module focusing on complex operator organization, GEMM optimization paths, property-based testing, and new architecture experiments.

## Prerequisites

Completion of `01-sgemm-tutorial` and `02-tensorcraft-core` is recommended.

## Contents

- GEMM optimization path (Naive → Pipeline)
- Reduction / Attention / Convolution / Quantization examples
- GoogleTest + RapidCheck testing
- Optional Python bindings
- CUDA 12/13 experimental features

## Sub-topics

- [GEMM Optimization](./03/gemm-optimization.md)
- [Memory Optimization](./03/memory-optimization.md)
- [Reduction Optimization](./03/reduction-optimization.md)
- [Flash Attention](./03/flash-attention.md)
- [CUDA 13 Features](./03/cuda13-features.md)

## References

[^1]: NVIDIA. "CUDA C++ Programming Guide," v12.6. 2024. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
[^2]: Dao, T., et al. "FlashAttention." *NeurIPS* 2022. https://arxiv.org/abs/2205.14135
[^3]: NVIDIA CUTLASS. https://github.com/NVIDIA/cutlass
