---
outline: [2, 3]
---

# 02-TensorCraft Core

TensorCraft Core is the teaching operator library of CUDA Kernel Academy. It uses a header-only design for easy reading and experimentation. It is **not** a production-grade kernel library: boundary handling, operator coverage, and performance are deliberately scoped for learning.

## Design Principles

```mermaid
graph TD
    A[Header-Only Design] --> B[Progressive Optimization]
    A --> C[Modern C++ First]
    B --> D[Backward Compatibility]
    C --> D

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
