---
outline: [2, 3]
---

# Bank Conflict Free SGEMM

Shared Memory is divided into 32 banks, each 4 bytes wide. Bank conflicts occur when multiple threads in the same warp access the same bank simultaneously.

## Padding Solution

```cpp
// Bank conflict
__shared__ float tile[32][32];

// No bank conflict (with padding)
__shared__ float tile[32][33];
```

## References
