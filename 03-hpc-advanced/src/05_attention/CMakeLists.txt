# Attention module
hpc_add_cuda_library(hpc_attention
    flash_attention.cu
    rope.cu
    topk.cu
)
