---
outline: [2, 3]
---

# GEMM Optimization

Detailed walkthrough of the 7-step GEMM optimization path.

## Optimization Path

```mermaid
flowchart LR
    Naive["Naive<br/>0.5 TFLOPS"] -->|"4×"| Tiled["Tiled<br/>2.1 TFLOPS"]
    Tiled -->|"2.1×"| Coalesced["Coalesced<br/>4.5 TFLOPS"]
    Coalesced -->|"1.8×"| DoubleBuffer["Double Buffer<br/>8.2 TFLOPS"]
    DoubleBuffer -->|"1.5×"| Vectorized["Vectorized<br/>12.5 TFLOPS"]
    Vectorized -->|"1.4×"| TensorCore["Tensor Core<br/>18.0 TFLOPS"]
    TensorCore -->|"1.17×"| CuBLAS["cuBLAS<br/>21.0 TFLOPS"]

    style Naive fill:#161b22,stroke:#76B900,color:#e6edf3
    style Tiled fill:#161b22,stroke:#76B900,color:#e6edf3
    style Coalesced fill:#161b22,stroke:#76B900,color:#e6edf3
    style DoubleBuffer fill:#161b22,stroke:#76B900,color:#e6edf3
    style Vectorized fill:#161b22,stroke:#76B900,color:#e6edf3
    style TensorCore fill:#161b22,stroke:#76B900,color:#e6edf3
    style CuBLAS fill:#161b22,stroke:#76B900,color:#e6edf3
```

## Summary

| Step | Technique | TFLOPS (FP32) | Speedup |
|------|-----------|---------------|---------|
| 1 | Naive | 0.5 | 1.0× |
| 2 | Shared Memory | 2.0 | 4.0× |
| 3 | Double Buffer | 3.5 | 7.0× |
| 4 | Register Tiling | 6.0 | 12.0× |
| 5 | WMMA | 50+ | 100× |
| 6 | MMA PTX | 60+ | 120× |
| 7 | Pipeline | 70+ | 140× |

## References

[^1]: NVIDIA. "CUDA C++ Programming Guide," v12.6. 2024. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
[^2]: Simon Boehm. "How to Optimize a CUDA Matmul Kernel." 2022. https://siboehm.com/articles/22/CUDA-MMM
[^3]: NVIDIA CUTLASS. https://github.com/NVIDIA/cutlass
