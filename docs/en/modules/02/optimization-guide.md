---
outline: [2, 3]
---

# Kernel Optimization Guide

This guide explains the optimization techniques used in TensorCraft Core kernels.

## GEMM Optimization Journey

```mermaid
flowchart LR
    Naive["Naive"] -->|"4×"| Tiled["Tiled"]
    Tiled -->|"1.75×"| DoubleBuffer["Double Buffer"]
    DoubleBuffer -->|"1.7×"| Register["Register Tiling"]
    Register -->|"8×"| TensorCore["Tensor Core"]
    TensorCore -->|"1.2×"| PTX["MMA PTX"]
    PTX -->|"1.15×"| Pipeline["Software Pipeline"]

    style Naive fill:#161b22,stroke:#76B900,color:#e6edf3
    style Tiled fill:#161b22,stroke:#76B900,color:#e6edf3
    style DoubleBuffer fill:#161b22,stroke:#76B900,color:#e6edf3
    style Register fill:#161b22,stroke:#76B900,color:#e6edf3
    style TensorCore fill:#161b22,stroke:#76B900,color:#e6edf3
    style PTX fill:#161b22,stroke:#76B900,color:#e6edf3
    style Pipeline fill:#161b22,stroke:#76B900,color:#e6edf3
```

## Key Techniques

- Shared Memory Tiling
- Double Buffering
- Register Blocking
- Tensor Core (WMMA)
- MMA PTX
- Software Pipelining

## References

[^1]: NVIDIA. "CUDA C++ Best Practices Guide." https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
[^2]: Simon Boehm. "How to Optimize a CUDA Matmul Kernel." 2022. https://siboehm.com/articles/22/CUDA-MMM
[^3]: NVIDIA CUTLASS. https://github.com/NVIDIA/cutlass
