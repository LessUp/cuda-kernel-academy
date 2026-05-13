---
outline: [2, 3]
---

# 02-TensorCraft Core

TensorCraft Core is the core operator library of CUDA Kernel Academy, providing production-grade CUDA kernel implementations. It uses a header-only design for easy integration.

## Design Principles

```mermaid
graph TD
    A[Header-Only Design] --> B[Progressive Optimization]
    A --> C[Modern C++ First]
    B --> D[Backward Compatibility]
    C --> D

    style A fill:#161b22,stroke:#76B900,color:#e6edf3
    style B fill:#161b22,stroke:#76B900,color:#e6edf3
    style C fill:#161b22,stroke:#76B900,color:#e6edf3
    style D fill:#161b22,stroke:#76B900,color:#e6edf3
```

## Core Features

| Feature | Description |
|---------|-------------|
| **Header-Only** | Zero compile dependencies |
| **Multi-Arch** | Volta (sm_70) to Hopper (sm_90) |
| **Python Bindings** | pybind11 Python interface |
| **Full Tests** | GoogleTest unit tests |
| **Benchmarks** | Google Benchmark performance tests |

## Quick Start

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure
```

## Sub-topics

- [Architecture](./02/architecture.md)
- [API Reference](./02/api-reference.md)
- [Optimization Guide](./02/optimization-guide.md)

## References

[^1]: NVIDIA. "CUDA C++ Programming Guide," v12.6. 2024. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
[^2]: NVIDIA CUTLASS. https://github.com/NVIDIA/cutlass
