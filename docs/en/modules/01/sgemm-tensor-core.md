---
outline: [2, 3]
---

# Tensor Core SGEMM

Use NVIDIA Tensor Cores for matrix multiplication via the WMMA API.

## WMMA Example

```cpp
#include <mma.h>
using namespace nvcuda;

wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

wmma::fill_fragment(c_frag, 0.0f);
wmma::load_matrix_sync(a_frag, A + offset, K);
wmma::load_matrix_sync(b_frag, B + offset, N);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(C + offset, c_frag, N, wmma::mem_row_major);
```

## Performance

- **Tensor Core throughput**: 8-16× higher than CUDA Cores
- **TFLOPS**: ~50+ (FP16)

## References
