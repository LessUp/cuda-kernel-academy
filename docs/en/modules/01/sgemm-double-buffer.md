---
outline: [2, 3]
---

# Double Buffer SGEMM

Use two sets of shared memory buffers to prefetch the next tile while computing the current tile.

## Key Idea

```cpp
__shared__ float As[2][TILE_SIZE][TILE_SIZE];
__shared__ float Bs[2][TILE_SIZE][TILE_SIZE];
```

## Performance

- **Hides memory latency**: computation overlaps with loading
- **TFLOPS**: ~3.5

## References
