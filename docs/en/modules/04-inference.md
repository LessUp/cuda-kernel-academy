---
outline: [2, 3]
---

# 04-Inference Engine

Lightweight CUDA inference engine demonstrating how to combine GEMM, Tensor, memory management, and simple weight file formats into a runnable inference pipeline.

## Contents

- `InferenceEngine`: Multi-layer linear network loading, execution, and timing
- `Tensor`: GPU Tensor wrapper and basic operators
- `MemoryPool`: Cached GPU memory pool
- `StreamManager`: Simple multi-stream manager
- `Config` / `Logger`: Configuration and logging utilities

## Build

When `04-inference-engine` is added from the repository root, it expects the
parent project to provide the existing `TensorCraft::tensorcraft` target. The
root `CMakeLists.txt` already enforces this by requiring `BUILD_TENSORCRAFT=ON`
before adding the inference module.

Standalone module builds still support the lightweight repository-layout
fallback: configuring from `04-inference-engine/` will try to add
`../02-tensorcraft-core` if it exists, and otherwise fall back to the local
GEMM implementations.

### Build from the repository root (recommended)

```bash
cmake --preset default
cmake --build --preset default
ctest --preset default
```

### Build the module standalone

```bash
cd 04-inference-engine
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

## Sub-topics

- [Quick Start](./04/quick-start.md)
- [Architecture](./04/architecture.md)
- [Performance Tuning](./04/performance-tuning.md)

## References

[^1]: NVIDIA. "CUDA C++ Programming Guide," v12.6. 2024. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
[^2]: NVIDIA cuBLAS. https://docs.nvidia.com/cuda/cublas/
