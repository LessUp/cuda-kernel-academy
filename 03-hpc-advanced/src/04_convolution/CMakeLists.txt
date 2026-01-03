# Convolution module
hpc_add_cuda_library(hpc_convolution
    conv_implicit_gemm.cu
    conv_winograd.cu
)
