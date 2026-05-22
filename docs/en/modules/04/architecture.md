---
outline: [2, 3]
---

# Inference Engine Architecture

Architecture design of the Mini-Inference Engine.

## System Architecture

```mermaid
flowchart TB
    subgraph App["Application Layer"]
        Benchmark["Benchmark"] --- MNIST["MNIST Demo"] --- Tests["Tests"]
    end

    subgraph Engine["Engine Layer"]
        IE["InferenceEngine"] --- Tensor["Tensor"] --- AutoTuner["AutoTuner"] --- Profiler["Profiler"]
    end

    subgraph Kernel["Kernel Layer"]
        N["Naive"] --- T["Tiled"] --- C["Coalesced"] --- D["Double Buffer"] --- O["Optimized"]
    end

    subgraph Infra["Infrastructure Layer"]
        MP["MemoryPool"] --- SM["StreamManager"] --- Logger["Logger"] --- Timer["GpuTimer"]
    end

    App --> Engine --> Kernel --> Infra

```

## Core Components

- **InferenceEngine**: Manages forward propagation
- **Tensor**: N-dimensional GPU tensor with basic ops
- **MemoryPool**: Caches GPU allocations
- **StreamManager**: CUDA stream management

## References
