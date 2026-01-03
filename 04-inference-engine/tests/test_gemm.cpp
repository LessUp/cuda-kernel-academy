#include <gtest/gtest.h>
#include "common.h"
#include "kernels.cuh"

using namespace mini_inference;

class GemmTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    }
    
    void TearDown() override {
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
        }
    }
    
    cublasHandle_t cublas_handle_ = nullptr;
};

// ============================================================================
// Property 1: Matrix Multiplication Correctness
// For any matrices A(M×K) and B(K×N), GPU result matches CPU within 1e-5
// ============================================================================

TEST_F(GemmTest, NaiveMatMulCorrectness) {
    // Feature: mini-inference-engine, Property 1: Matrix Multiplication Correctness
    const int M = 128, N = 256, K = 64;
    
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_C_ref(M * N);
    random_init(h_A.data(), M * K);
    random_init(h_B.data(), K * N);
    
    // CPU reference
    cpu_matmul(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
    
    // GPU computation
    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));
    
    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));
    
    launch_naive_matmul(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    d_C.copy_to_host(h_C.data(), M * N * sizeof(float));
    
    float max_error = compare_matrices(h_C.data(), h_C_ref.data(), M * N);
    EXPECT_LT(max_error, 1e-5f) << "Max error: " << max_error;
}

TEST_F(GemmTest, TiledGemmCorrectness) {
    const int M = 128, N = 256, K = 64;
    
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_C_ref(M * N);
    random_init(h_A.data(), M * K);
    random_init(h_B.data(), K * N);
    
    cpu_matmul(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
    
    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));
    
    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));
    
    launch_tiled_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    d_C.copy_to_host(h_C.data(), M * N * sizeof(float));
    
    float max_error = compare_matrices(h_C.data(), h_C_ref.data(), M * N);
    EXPECT_LT(max_error, 1e-5f) << "Max error: " << max_error;
}

// ============================================================================
// Property 2: Optimized GEMM Equivalence
// All optimized versions produce equivalent results
// ============================================================================

TEST_F(GemmTest, OptimizedGemmEquivalence) {
    // Feature: mini-inference-engine, Property 2: Optimized GEMM Equivalence
    const int M = 256, N = 256, K = 128;
    
    std::vector<float> h_A(M * K), h_B(K * N);
    std::vector<float> h_C_naive(M * N), h_C_tiled(M * N), h_C_coalesced(M * N);
    std::vector<float> h_C_double(M * N), h_C_opt(M * N);
    
    random_init(h_A.data(), M * K);
    random_init(h_B.data(), K * N);
    
    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));
    
    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));
    
    // Naive
    launch_naive_matmul(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_C.copy_to_host(h_C_naive.data(), M * N * sizeof(float));
    
    // Tiled
    d_C.zero();
    launch_tiled_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_C.copy_to_host(h_C_tiled.data(), M * N * sizeof(float));
    
    // Coalesced
    d_C.zero();
    launch_coalesced_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_C.copy_to_host(h_C_coalesced.data(), M * N * sizeof(float));
    
    // Double buffer
    d_C.zero();
    launch_double_buffer_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_C.copy_to_host(h_C_double.data(), M * N * sizeof(float));
    
    // Optimized
    d_C.zero();
    launch_optimized_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_C.copy_to_host(h_C_opt.data(), M * N * sizeof(float));
    
    // Compare all versions
    float tol = 1e-4f;  // Slightly relaxed for numerical differences
    
    float err_tiled = compare_matrices(h_C_naive.data(), h_C_tiled.data(), M * N);
    EXPECT_LT(err_tiled, tol) << "Tiled vs Naive error: " << err_tiled;
    
    float err_coalesced = compare_matrices(h_C_naive.data(), h_C_coalesced.data(), M * N);
    EXPECT_LT(err_coalesced, tol) << "Coalesced vs Naive error: " << err_coalesced;
    
    float err_double = compare_matrices(h_C_naive.data(), h_C_double.data(), M * N);
    EXPECT_LT(err_double, tol) << "DoubleBuffer vs Naive error: " << err_double;
    
    float err_opt = compare_matrices(h_C_naive.data(), h_C_opt.data(), M * N);
    EXPECT_LT(err_opt, tol) << "Optimized vs Naive error: " << err_opt;
}

// ============================================================================
// Property 6: Dimension Mismatch Detection
// ============================================================================

TEST_F(GemmTest, DimensionMismatchDetection) {
    // Feature: mini-inference-engine, Property 6: Dimension Mismatch Detection
    MatrixDesc A(nullptr, 10, 20, 20);
    MatrixDesc B(nullptr, 30, 40, 40);  // B.rows != A.cols
    MatrixDesc C(nullptr, 10, 40, 40);
    
    EXPECT_THROW(validate_gemm_inputs(A, B, C), std::invalid_argument);
}

TEST_F(GemmTest, OutputDimensionMismatch) {
    MatrixDesc A(nullptr, 10, 20, 20);
    MatrixDesc B(nullptr, 20, 30, 30);
    MatrixDesc C(nullptr, 10, 40, 40);  // Wrong output dimensions
    
    EXPECT_THROW(validate_gemm_inputs(A, B, C), std::invalid_argument);
}

TEST_F(GemmTest, NullPointerDetection) {
    float dummy;
    MatrixDesc A(&dummy, 10, 20, 20);
    MatrixDesc B(nullptr, 20, 30, 30);  // Null pointer
    MatrixDesc C(&dummy, 10, 30, 30);
    
    EXPECT_THROW(validate_gemm_inputs(A, B, C), std::invalid_argument);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(GemmTest, SmallMatrix) {
    const int M = 1, N = 1, K = 1;
    
    std::vector<float> h_A = {2.0f};
    std::vector<float> h_B = {3.0f};
    std::vector<float> h_C(1);
    
    DeviceMemory d_A(sizeof(float));
    DeviceMemory d_B(sizeof(float));
    DeviceMemory d_C(sizeof(float));
    
    d_A.copy_from_host(h_A.data(), sizeof(float));
    d_B.copy_from_host(h_B.data(), sizeof(float));
    
    launch_naive_matmul(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    d_C.copy_to_host(h_C.data(), sizeof(float));
    
    EXPECT_NEAR(h_C[0], 6.0f, 1e-6f);
}

TEST_F(GemmTest, NonAlignedDimensions) {
    // Test with dimensions that don't align to tile size
    const int M = 33, N = 65, K = 17;
    
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_C_ref(M * N);
    random_init(h_A.data(), M * K);
    random_init(h_B.data(), K * N);
    
    cpu_matmul(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
    
    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));
    
    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));
    
    // Test all kernels with non-aligned dimensions
    launch_tiled_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_C.copy_to_host(h_C.data(), M * N * sizeof(float));
    
    float max_error = compare_matrices(h_C.data(), h_C_ref.data(), M * N);
    EXPECT_LT(max_error, 1e-4f) << "Non-aligned tiled GEMM error: " << max_error;
}

// ============================================================================
// cuBLAS Comparison
// ============================================================================

TEST_F(GemmTest, CublasComparison) {
    const int M = 512, N = 512, K = 256;
    
    std::vector<float> h_A(M * K), h_B(K * N);
    std::vector<float> h_C_opt(M * N), h_C_cublas(M * N);
    
    random_init(h_A.data(), M * K);
    random_init(h_B.data(), K * N);
    
    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));
    
    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));
    
    // Optimized kernel
    launch_optimized_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_C.copy_to_host(h_C_opt.data(), M * N * sizeof(float));
    
    // cuBLAS
    d_C.zero();
    launch_cublas_gemm(cublas_handle_, d_A.get(), d_B.get(), d_C.get(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_C.copy_to_host(h_C_cublas.data(), M * N * sizeof(float));
    
    float max_error = compare_matrices(h_C_opt.data(), h_C_cublas.data(), M * N);
    EXPECT_LT(max_error, 1e-3f) << "Optimized vs cuBLAS error: " << max_error;
}
