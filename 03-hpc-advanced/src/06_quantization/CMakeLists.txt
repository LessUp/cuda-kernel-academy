# Quantization module
hpc_add_cuda_library(hpc_quantization
    dequant.cu
    int8_quant.cu
    fp8_scaling.cu
)
