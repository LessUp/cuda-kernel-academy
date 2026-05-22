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

```

## Key Techniques

- Shared Memory Tiling
- Double Buffering
- Register Blocking
- Tensor Core (WMMA)
- MMA PTX
- Software Pipelining

## References
