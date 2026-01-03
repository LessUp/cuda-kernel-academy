#pragma once

#include <cuda_runtime.h>

namespace hpc::cuda13 {

struct ClusterConfig {
    dim3 cluster_dims;  // e.g., {2, 1, 1} for 2-block cluster
    dim3 grid_dims;
    dim3 block_dims;
};

template <typename T>
void cluster_reduce(const T* input, T* output, size_t n,
                    const ClusterConfig& config,
                    cudaStream_t stream = nullptr);

} // namespace hpc::cuda13
