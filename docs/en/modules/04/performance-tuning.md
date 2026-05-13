---
outline: [2, 3]
---

# Performance Tuning

Performance tuning guide for the Mini-Inference Engine.

## Key Metrics

1. **GFLOPS**: FLOPs / (time_ms * 1e6)
2. **Memory Bandwidth Utilization**: Actual / Peak bandwidth
3. **Arithmetic Intensity**: FLOPs / Bytes

## Block Size Selection

| Parameter | Description | Typical |
|-----------|-------------|---------|
| BM | Block M dimension | 64-256 |
| BN | Block N dimension | 64-256 |
| BK | Iteration K dimension | 8-32 |
| TM | Thread M dimension | 4-8 |
| TN | Thread N dimension | 4-8 |

## Architecture-Specific Tips

- **Volta (SM 7.0)**: BM=128, BN=128
- **Turing (SM 7.5)**: Consider FP16
- **Ampere (SM 8.0)**: Use `cp.async`
- **Ada (SM 8.9)**: Maximize compute density

## Profiling Tools

```bash
nsys profile -o report ./benchmark
ncu --set full -o report ./benchmark
```

## References

[^1]: NVIDIA. "Nsight Compute Documentation." https://docs.nvidia.com/nsight-compute/
[^2]: NVIDIA. "CUDA C++ Best Practices Guide." https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
