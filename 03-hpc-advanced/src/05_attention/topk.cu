#include "topk.cuh"
#include "../common/cuda_check.cuh"

namespace hpc::attention {

// Simple bitonic sort based TopK for small k
template <typename T>
__global__ void topk_kernel(const T* __restrict__ input,
                            T* __restrict__ output,
                            int* __restrict__ indices,
                            int n, int k) {
    int batch_idx = blockIdx.x;
    const T* batch_input = input + batch_idx * n;
    T* batch_output = output + batch_idx * k;
    int* batch_indices = indices + batch_idx * k;

    // Use shared memory for partial sorting
    extern __shared__ char smem[];
    T* vals = reinterpret_cast<T*>(smem);
    int* idxs = reinterpret_cast<int*>(vals + blockDim.x);

    // Initialize with first elements
    int tid = threadIdx.x;
    if (tid < n) {
        vals[tid] = batch_input[tid];
        idxs[tid] = tid;
    } else {
        vals[tid] = -1e30f;  // Very small value
        idxs[tid] = -1;
    }
    __syncthreads();

    // Simple selection: find k largest
    for (int i = 0; i < k; ++i) {
        // Find max in remaining elements
        if (tid == 0) {
            int max_idx = i;
            T max_val = vals[i];
            for (int j = i + 1; j < n && j < blockDim.x; ++j) {
                if (vals[j] > max_val) {
                    max_val = vals[j];
                    max_idx = j;
                }
            }
            // Swap
            T tmp_val = vals[i];
            int tmp_idx = idxs[i];
            vals[i] = vals[max_idx];
            idxs[i] = idxs[max_idx];
            vals[max_idx] = tmp_val;
            idxs[max_idx] = tmp_idx;
        }
        __syncthreads();
    }

    // Write output
    if (tid < k) {
        batch_output[tid] = vals[tid];
        batch_indices[tid] = idxs[tid];
    }
}

template <>
void topk<float>(const float* input, float* output, int* indices,
                 int batch, int n, int k, cudaStream_t stream) {
    int block_size = min(n, 1024);
    size_t smem_size = block_size * (sizeof(float) + sizeof(int));
    topk_kernel<float><<<batch, block_size, smem_size, stream>>>(
        input, output, indices, n, k);
    CUDA_CHECK_LAST();
}

} // namespace hpc::attention
