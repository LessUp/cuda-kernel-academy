---
outline: [2, 3]
---

# CUDA 13 & Hopper Features

::: warning Performance numbers are teaching placeholders
The TFLOPS, speedup and memory figures on this page are reference values, not results reproduced by this repository on a pinned hardware/software stack. Re-measure on your own GPU before quoting them.
:::

New features in the CUDA 12/13 era related to the Hopper architecture.

## Key Features

1. **TMA (Tensor Memory Accelerator)**: Async data movement
2. **Thread Block Clusters**: Inter-block cooperation
3. **Distributed Shared Memory**: Cross-block shared memory
4. **FP8 Data Types**: e4m3 and e5m2
5. **Asynchronous Transaction Barrier**: New sync primitive

## Hopper Improvements

| Feature | Ampere (A100) | Hopper (H100) | Improvement |
|---------|---------------|---------------|-------------|
| FP16 Tensor Core | 312 TFLOPS | 989 TFLOPS | 3.2× |
| FP8 Tensor Core | - | 1979 TFLOPS | New |
| HBM Bandwidth | 2 TB/s | 3.35 TB/s | 1.7× |
| Shared Memory | 164 KB/SM | 228 KB/SM | 1.4× |

## References
