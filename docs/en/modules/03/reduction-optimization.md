---
outline: [2, 3]
---

# Reduction Optimization

CUDA reduction optimization techniques including Warp Shuffle, Block Reduce, and Online Softmax.

## Warp Shuffle Reduction

```cpp
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

## Online Softmax

Single-pass softmax maintaining running max and sum:

```cpp
float max_val = -INFINITY, sum = 0.0f;
for (int i = 0; i < n; ++i) {
    float x = input[i];
    float old_max = max_val;
    max_val = fmaxf(max_val, x);
    sum = sum * expf(old_max - max_val) + expf(x - max_val);
}
```

## References
