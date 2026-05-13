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

    style App fill:#161b22,stroke:#76B900,color:#e6edf3
    style Engine fill:#161b22,stroke:#76B900,color:#e6edf3
    style Kernel fill:#161b22,stroke:#76B900,color:#e6edf3
    style Infra fill:#161b22,stroke:#76B900,color:#e6edf3
```

## Core Components

- **InferenceEngine**: Manages forward propagation
- **Tensor**: N-dimensional GPU tensor with basic ops
- **MemoryPool**: Caches GPU allocations
- **StreamManager**: CUDA stream management

## References

[^1]: NVIDIA. "CUDA C++ Programming Guide," v12.6. 2024. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
[^2]: NVIDIA cuBLAS. https://docs.nvidia.com/cuda/cublas/
