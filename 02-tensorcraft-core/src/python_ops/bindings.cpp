/**
 * @file bindings.cpp
 * @brief Python bindings for TensorCraft-HPC using pybind11
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "tensorcraft/core/cuda_check.hpp"
#include "tensorcraft/memory/tensor.hpp"
#include "tensorcraft/kernels/elementwise.hpp"
#include "tensorcraft/kernels/softmax.hpp"
#include "tensorcraft/kernels/normalization.hpp"
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/kernels/attention.hpp"

namespace py = pybind11;
using namespace tensorcraft;

// Helper to convert numpy array to device pointer
template<typename T>
T* numpy_to_device(py::array_t<T> arr, size_t& size) {
    py::buffer_info buf = arr.request();
    size = buf.size;
    
    T* d_ptr;
    TC_CUDA_CHECK(cudaMalloc(&d_ptr, size * sizeof(T)));
    TC_CUDA_CHECK(cudaMemcpy(d_ptr, buf.ptr, size * sizeof(T), cudaMemcpyHostToDevice));
    return d_ptr;
}

// Helper to copy device data to numpy array
template<typename T>
py::array_t<T> device_to_numpy(T* d_ptr, const std::vector<ssize_t>& shape) {
    size_t size = 1;
    for (auto s : shape) size *= s;
    
    py::array_t<T> result(shape);
    py::buffer_info buf = result.request();
    TC_CUDA_CHECK(cudaMemcpy(buf.ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
}

// RAII wrapper for device memory
template<typename T>
struct DeviceArray {
    T* ptr = nullptr;
    size_t size = 0;
    
    DeviceArray(py::array_t<T> arr) {
        ptr = numpy_to_device(arr, size);
    }
    
    ~DeviceArray() {
        if (ptr) cudaFree(ptr);
    }
    
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

// ============================================================================
// Elementwise Operations
// ============================================================================

py::array_t<float> py_relu(py::array_t<float> input) {
    py::buffer_info buf = input.request();
    size_t n = buf.size;
    
    DeviceArray<float> d_input(input);
    float* d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    
    kernels::relu(d_input.ptr, d_output, n);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<ssize_t> shape(buf.shape.begin(), buf.shape.end());
    auto result = device_to_numpy(d_output, shape);
    cudaFree(d_output);
    return result;
}

py::array_t<float> py_silu(py::array_t<float> input) {
    py::buffer_info buf = input.request();
    size_t n = buf.size;
    
    DeviceArray<float> d_input(input);
    float* d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    
    kernels::silu(d_input.ptr, d_output, n);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<ssize_t> shape(buf.shape.begin(), buf.shape.end());
    auto result = device_to_numpy(d_output, shape);
    cudaFree(d_output);
    return result;
}

py::array_t<float> py_gelu(py::array_t<float> input) {
    py::buffer_info buf = input.request();
    size_t n = buf.size;
    
    DeviceArray<float> d_input(input);
    float* d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    
    kernels::gelu(d_input.ptr, d_output, n);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<ssize_t> shape(buf.shape.begin(), buf.shape.end());
    auto result = device_to_numpy(d_output, shape);
    cudaFree(d_output);
    return result;
}

py::array_t<float> py_vector_add(py::array_t<float> a, py::array_t<float> b) {
    py::buffer_info buf_a = a.request();
    py::buffer_info buf_b = b.request();
    
    if (buf_a.size != buf_b.size) {
        throw std::runtime_error("Input arrays must have the same size");
    }
    
    size_t n = buf_a.size;
    DeviceArray<float> d_a(a);
    DeviceArray<float> d_b(b);
    float* d_c;
    TC_CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));
    
    kernels::vector_add(d_a.ptr, d_b.ptr, d_c, n);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<ssize_t> shape(buf_a.shape.begin(), buf_a.shape.end());
    auto result = device_to_numpy(d_c, shape);
    cudaFree(d_c);
    return result;
}

// ============================================================================
// Softmax
// ============================================================================

py::array_t<float> py_softmax(py::array_t<float> input) {
    py::buffer_info buf = input.request();
    
    if (buf.ndim < 1) {
        throw std::runtime_error("Input must have at least 1 dimension");
    }
    
    int cols = buf.shape[buf.ndim - 1];
    int rows = buf.size / cols;
    
    DeviceArray<float> d_input(input);
    float* d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_output, buf.size * sizeof(float)));
    
    kernels::softmax(d_input.ptr, d_output, rows, cols);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<ssize_t> shape(buf.shape.begin(), buf.shape.end());
    auto result = device_to_numpy(d_output, shape);
    cudaFree(d_output);
    return result;
}

// ============================================================================
// Normalization
// ============================================================================

py::array_t<float> py_layernorm(
    py::array_t<float> input,
    py::array_t<float> gamma,
    py::array_t<float> beta,
    float eps = 1e-5f) {
    
    py::buffer_info buf = input.request();
    if (buf.ndim < 2) {
        throw std::runtime_error("Input must have at least 2 dimensions");
    }
    
    int hidden_size = buf.shape[buf.ndim - 1];
    int batch_size = buf.size / hidden_size;
    
    DeviceArray<float> d_input(input);
    DeviceArray<float> d_gamma(gamma);
    DeviceArray<float> d_beta(beta);
    float* d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_output, buf.size * sizeof(float)));
    
    kernels::layernorm(d_input.ptr, d_gamma.ptr, d_beta.ptr, d_output,
                       batch_size, hidden_size, eps);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<ssize_t> shape(buf.shape.begin(), buf.shape.end());
    auto result = device_to_numpy(d_output, shape);
    cudaFree(d_output);
    return result;
}

py::array_t<float> py_rmsnorm(
    py::array_t<float> input,
    py::array_t<float> weight,
    float eps = 1e-6f) {
    
    py::buffer_info buf = input.request();
    if (buf.ndim < 2) {
        throw std::runtime_error("Input must have at least 2 dimensions");
    }
    
    int hidden_size = buf.shape[buf.ndim - 1];
    int batch_size = buf.size / hidden_size;
    
    DeviceArray<float> d_input(input);
    DeviceArray<float> d_weight(weight);
    float* d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_output, buf.size * sizeof(float)));
    
    kernels::rmsnorm(d_input.ptr, d_weight.ptr, d_output,
                     batch_size, hidden_size, eps);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<ssize_t> shape(buf.shape.begin(), buf.shape.end());
    auto result = device_to_numpy(d_output, shape);
    cudaFree(d_output);
    return result;
}

// ============================================================================
// GEMM
// ============================================================================

py::array_t<float> py_gemm(
    py::array_t<float> A,
    py::array_t<float> B,
    float alpha = 1.0f,
    float beta = 0.0f,
    const std::string& version = "tiled") {
    
    py::buffer_info buf_a = A.request();
    py::buffer_info buf_b = B.request();
    
    if (buf_a.ndim != 2 || buf_b.ndim != 2) {
        throw std::runtime_error("A and B must be 2D matrices");
    }
    
    int M = buf_a.shape[0];
    int K = buf_a.shape[1];
    int N = buf_b.shape[1];
    
    if (buf_b.shape[0] != K) {
        throw std::runtime_error("Matrix dimensions don't match for multiplication");
    }
    
    DeviceArray<float> d_A(A);
    DeviceArray<float> d_B(B);
    float* d_C;
    TC_CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    TC_CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
    
    kernels::GemmVersion ver = kernels::GemmVersion::Tiled;
    if (version == "naive") ver = kernels::GemmVersion::Naive;
    else if (version == "double_buffer") ver = kernels::GemmVersion::DoubleBuffer;
    
    kernels::launch_gemm(d_A.ptr, d_B.ptr, d_C, M, N, K, alpha, beta, ver);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    
    auto result = device_to_numpy(d_C, {M, N});
    cudaFree(d_C);
    return result;
}

// ============================================================================
// Module Definition
// ============================================================================

PYBIND11_MODULE(tensorcraft_ops, m) {
    m.doc() = "TensorCraft-HPC: High-Performance AI Kernels";
    
    // Elementwise operations
    m.def("relu", &py_relu, "ReLU activation",
          py::arg("input"));
    m.def("silu", &py_silu, "SiLU (Swish) activation",
          py::arg("input"));
    m.def("gelu", &py_gelu, "GeLU activation",
          py::arg("input"));
    m.def("vector_add", &py_vector_add, "Element-wise vector addition",
          py::arg("a"), py::arg("b"));
    
    // Softmax
    m.def("softmax", &py_softmax, "Softmax along last dimension",
          py::arg("input"));
    
    // Normalization
    m.def("layernorm", &py_layernorm, "Layer normalization",
          py::arg("input"), py::arg("gamma"), py::arg("beta"),
          py::arg("eps") = 1e-5f);
    m.def("rmsnorm", &py_rmsnorm, "RMS normalization",
          py::arg("input"), py::arg("weight"),
          py::arg("eps") = 1e-6f);
    
    // GEMM
    m.def("gemm", &py_gemm, "General matrix multiplication",
          py::arg("A"), py::arg("B"),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          py::arg("version") = "tiled");
    
    // Version info
    m.attr("__version__") = "0.1.0";
}
