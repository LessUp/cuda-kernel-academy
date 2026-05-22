---
outline: [2, 3]
---

# Memory Optimization

CUDA memory optimization techniques including coalesced access, vectorized loads, and shared memory usage.

## Coalesced Access

When a warp (32 threads) accesses consecutive memory addresses, the GPU can merge these into one or few memory transactions.

## Vectorized Loads

```cpp
// Scalar: 4 memory transactions
float a = input[idx];
float b = input[idx+1];
float c = input[idx+2];
float d = input[idx+3];

// Vectorized: 1 memory transaction
float4 vec = *reinterpret_cast<const float4*>(&input[idx]);
```

## Bank Conflict Avoidance

```cpp
// Bank conflict
__shared__ float tile[32][32];

// No bank conflict (padding)
__shared__ float tile[32][33];
```

## References
