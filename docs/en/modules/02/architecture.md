---
outline: [2, 3]
---

# TensorCraft Core Architecture

::: warning Teaching architecture
This page describes a teaching design. Headers and tests are authoritative.
:::


This document describes the overall architecture and design decisions of TensorCraft Core.

## Module Architecture

```mermaid
flowchart TB
    subgraph tensorcraft["tensorcraft/"]
        core["core/"] --> kernels["kernels/"]
        memory["memory/"] --> kernels
    end

```

## Core Layer

Provides infrastructure: `cuda_check.hpp`, `features.hpp`, `type_traits.hpp`.

## Memory Layer

Provides memory abstractions: `AlignedVector`, `Tensor`, `MemoryPool`.

## Kernels Layer

Operator implementations: GEMM, Attention, Normalization, Conv2D, Sparse, Fusion.

## References
