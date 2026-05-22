---
outline: [2, 3]
---

# Naive SGEMM

Each thread computes one element of the output matrix C. This is the minimal starting point for understanding CUDA GEMM.

## Implementation

```cpp
__global__ void gemm_naive_kernel(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

## Performance

- **Issue**: Each element requires 2K global memory accesses
- **Bandwidth utilization**: ~5-10%
- **TFLOPS**: ~0.5 (FP32, RTX 4090)

## References
