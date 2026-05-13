---
outline: [2, 3]
---

# TensorCraft Core Architecture

This document describes the overall architecture and design decisions of TensorCraft Core.

## Module Architecture

```mermaid
flowchart TB
    subgraph tensorcraft["tensorcraft/"]
        core["core/"] --> kernels["kernels/"]
        memory["memory/"] --> kernels
    end

    style tensorcraft fill:#161b22,stroke:#76B900,color:#e6edf3
    style core fill:#161b22,stroke:#76B900,color:#e6edf3
    style memory fill:#161b22,stroke:#76B900,color:#e6edf3
    style kernels fill:#161b22,stroke:#76B900,color:#e6edf3
```

## Core Layer

Provides infrastructure: `cuda_check.hpp`, `features.hpp`, `type_traits.hpp`.

## Memory Layer

Provides memory abstractions: `AlignedVector`, `Tensor`, `MemoryPool`.

## Kernels Layer

Operator implementations: GEMM, Attention, Normalization, Conv2D, Sparse, Fusion.

## References

[^1]: NVIDIA. "CUDA C++ Programming Guide," v12.6. 2024. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
[^2]: NVIDIA CUTLASS. https://github.com/NVIDIA/cutlass
