// HPC-AI-Optimization-Lab Python Bindings
// Using Nanobind for zero-copy PyTorch tensor integration

#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>
#include <cuda_runtime.h>

// Include kernel headers
#include "01_elementwise/relu.cuh"
#include "01_elementwise/sigmoid.cuh"
#include "01_elementwise/transpose.cuh"
#include "01_elementwise/vector_add.cuh"
#include "02_reduction/softmax.cuh"
#include "02_reduction/layernorm.cuh"
#include "02_reduction/rmsnorm.cuh"
#include "03_gemm/gemm.cuh"
#include "05_attention/flash_attention.cuh"
#include "05_attention/rope.cuh"

namespace nb = nanobind;

// Helper to get CUDA pointer from PyTorch tensor
template<typename T>
T* get_cuda_ptr(nb::tensor<T, nb::device::cuda>& tensor) {
    return tensor.data();
}

// Elementwise operations
void relu_wrapper(nb::tensor<float, nb::device::cuda>& input,
                  nb::tensor<float, nb::device::cuda>& output) {
    size_t n = input.size();
    hpc::elementwise::relu<float, hpc::elementwise::OptLevel::GridStride>(
        input.data(), output.data(), n, nullptr);
    cudaDeviceSynchronize();
}

void sigmoid_wrapper(nb::tensor<float, nb::device::cuda>& input,
                     nb::tensor<float, nb::device::cuda>& output) {
    size_t n = input.size();
    hpc::elementwise::sigmoid<float, hpc::elementwise::OptLevel::GridStride>(
        input.data(), output.data(), n, nullptr);
    cudaDeviceSynchronize();
}

void transpose_wrapper(nb::tensor<float, nb::device::cuda>& input,
                       nb::tensor<float, nb::device::cuda>& output,
                       int rows, int cols) {
    hpc::elementwise::transpose<float, hpc::elementwise::TransposeOpt::SharedMemPadded>(
        input.data(), output.data(), rows, cols, nullptr);
    cudaDeviceSynchronize();
}

// Reduction operations
void softmax_wrapper(nb::tensor<float, nb::device::cuda>& input,
                     nb::tensor<float, nb::device::cuda>& output,
                     int batch, int seq_len) {
    hpc::reduction::softmax<float, hpc::reduction::SoftmaxOpt::OnlineSoftmax>(
        input.data(), output.data(), batch, seq_len, nullptr);
    cudaDeviceSynchronize();
}

void layer_norm_wrapper(nb::tensor<float, nb::device::cuda>& input,
                        nb::tensor<float, nb::device::cuda>& gamma,
                        nb::tensor<float, nb::device::cuda>& beta,
                        nb::tensor<float, nb::device::cuda>& output,
                        int batch, int hidden_size, float eps) {
    hpc::reduction::layer_norm<float>(
        input.data(), gamma.data(), beta.data(), output.data(),
        batch, hidden_size, eps, nullptr);
    cudaDeviceSynchronize();
}

void rms_norm_wrapper(nb::tensor<float, nb::device::cuda>& input,
                      nb::tensor<float, nb::device::cuda>& gamma,
                      nb::tensor<float, nb::device::cuda>& output,
                      int batch, int hidden_size, float eps) {
    hpc::reduction::rms_norm<float>(
        input.data(), gamma.data(), output.data(),
        batch, hidden_size, eps, nullptr);
    cudaDeviceSynchronize();
}

// GEMM
void matmul_wrapper(nb::tensor<float, nb::device::cuda>& A,
                    nb::tensor<float, nb::device::cuda>& B,
                    nb::tensor<float, nb::device::cuda>& C,
                    int M, int N, int K,
                    float alpha, float beta) {
    hpc::gemm::gemm<float, hpc::gemm::GemmOpt::SharedMemTiling>(
        A.data(), B.data(), C.data(), M, N, K, alpha, beta, nullptr);
    cudaDeviceSynchronize();
}

NB_MODULE(hpc_kernels, m) {
    m.doc() = "HPC-AI-Optimization-Lab CUDA Kernels";
    
    // Elementwise submodule
    auto elementwise = m.def_submodule("elementwise", "Elementwise operations");
    elementwise.def("relu", &relu_wrapper, "ReLU activation");
    elementwise.def("sigmoid", &sigmoid_wrapper, "Sigmoid activation");
    elementwise.def("transpose", &transpose_wrapper, "Matrix transpose");
    
    // Reduction submodule
    auto reduction = m.def_submodule("reduction", "Reduction operations");
    reduction.def("softmax", &softmax_wrapper, "Softmax");
    reduction.def("layer_norm", &layer_norm_wrapper, "Layer normalization");
    reduction.def("rms_norm", &rms_norm_wrapper, "RMS normalization");
    
    // GEMM submodule
    auto gemm = m.def_submodule("gemm", "Matrix multiplication");
    gemm.def("matmul", &matmul_wrapper, "Matrix multiplication");
}
