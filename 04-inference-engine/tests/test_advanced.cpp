#include <gtest/gtest.h>
#include "common.h"
#include "kernels.cuh"
#include "vectorized_gemm.cuh"
#include "half_gemm.cuh"
#include "autotuner.h"
#include "profiler.h"

using namespace mini_inference;

class AdvancedTest : public ::testing::Test {
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
// Vectorized GEMM Tests
// ============================================================================

TEST_F(AdvancedTest, VectorizedGemmCorrectness) {
    const int M = 256, N = 256, K = 128;
    
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_C_ref(M * N);
    random_init(h_A.data(), M * K);
    random_init(h_B.data(), K * N);
    
    // CPU reference
    cpu_matmul(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
    
    // GPU vectorized
    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));
    
    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));
    
    launch_vectorized_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    d_C.copy_to_host(h_C.data(), M * N * sizeof(float));
    
    float max_error = compare_matrices(h_C.data(), h_C_ref.data(), M * N);
    EXPECT_LT(max_error, 1e-4f) << "Vectorized GEMM error: " << max_error;
}

TEST_F(AdvancedTest, VectorizedGemmLargeMatrix) {
    const int M = 1024, N = 1024, K = 512;
    
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_C_ref(M * N);
    random_init(h_A.data(), M * K);
    random_init(h_B.data(), K * N);
    
    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));
    DeviceMemory d_C_ref(M * N * sizeof(float));
    
    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));
    
    // Compare with cuBLAS
    launch_cublas_gemm(cublas_handle_, d_A.get(), d_B.get(), d_C_ref.get(), M, N, K);
    launch_vectorized_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    d_C.copy_to_host(h_C.data(), M * N * sizeof(float));
    d_C_ref.copy_to_host(h_C_ref.data(), M * N * sizeof(float));
    
    float max_error = compare_matrices(h_C.data(), h_C_ref.data(), M * N);
    EXPECT_LT(max_error, 1e-3f) << "Vectorized vs cuBLAS error: " << max_error;
}

// ============================================================================
// Half Precision Tests
// ============================================================================

TEST_F(AdvancedTest, HalfPrecisionConversion) {
    const int N = 1024;
    
    std::vector<float> h_float(N);
    random_init(h_float.data(), N);
    
    DeviceMemory d_float(N * sizeof(float));
    DeviceMemory d_half_mem(N * sizeof(half));
    DeviceMemory d_float_back(N * sizeof(float));
    
    d_float.copy_from_host(h_float.data(), N * sizeof(float));
    
    // Convert float -> half -> float
    convert_float_to_half(d_float.get(), reinterpret_cast<half*>(d_half_mem.get()), N);
    convert_half_to_float(reinterpret_cast<half*>(d_half_mem.get()), d_float_back.get(), N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<float> h_float_back(N);
    d_float_back.copy_to_host(h_float_back.data(), N * sizeof(float));
    
    // FP16 has limited precision, allow larger error
    float max_error = compare_matrices(h_float.data(), h_float_back.data(), N);
    EXPECT_LT(max_error, 1e-2f) << "FP16 conversion error: " << max_error;
}

TEST_F(AdvancedTest, MixedPrecisionGemm) {
    const int M = 128, N = 128, K = 64;
    
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_C_ref(M * N);
    random_init(h_A.data(), M * K, -0.5f, 0.5f);  // Smaller range for FP16
    random_init(h_B.data(), K * N, -0.5f, 0.5f);
    
    // CPU reference
    cpu_matmul(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
    
    // Convert to half
    DeviceMemory d_A_float(M * K * sizeof(float));
    DeviceMemory d_B_float(K * N * sizeof(float));
    DeviceMemory d_A_half(M * K * sizeof(half));
    DeviceMemory d_B_half(K * N * sizeof(half));
    DeviceMemory d_C(M * N * sizeof(float));
    
    d_A_float.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B_float.copy_from_host(h_B.data(), K * N * sizeof(float));
    
    convert_float_to_half(d_A_float.get(), reinterpret_cast<half*>(d_A_half.get()), M * K);
    convert_float_to_half(d_B_float.get(), reinterpret_cast<half*>(d_B_half.get()), K * N);
    
    // Mixed precision GEMM
    launch_mixed_precision_gemm(
        reinterpret_cast<half*>(d_A_half.get()),
        reinterpret_cast<half*>(d_B_half.get()),
        d_C.get(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    d_C.copy_to_host(h_C.data(), M * N * sizeof(float));
    
    // FP16 accumulation has lower precision
    float max_error = compare_matrices(h_C.data(), h_C_ref.data(), M * N);
    EXPECT_LT(max_error, 0.1f) << "Mixed precision GEMM error: " << max_error;
}

// ============================================================================
// Auto-Tuner Tests
// ============================================================================

TEST_F(AdvancedTest, AutoTunerBasic) {
    AutoTuner tuner;
    
    // Tune for a specific size
    auto result = tuner.tune(512, 512, 256, cublas_handle_);
    
    EXPECT_GT(result.gflops, 0.0f);
    EXPECT_GT(result.time_ms, 0.0f);
    
    // Verify cached result
    auto cached = tuner.get_best(512, 512, 256, cublas_handle_);
    EXPECT_EQ(cached.config.kernel_type, result.config.kernel_type);
}

TEST_F(AdvancedTest, AutoTunerExecution) {
    AutoTuner tuner;
    
    const int M = 256, N = 256, K = 128;
    
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_C_ref(M * N);
    random_init(h_A.data(), M * K);
    random_init(h_B.data(), K * N);
    
    cpu_matmul(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
    
    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));
    
    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));
    
    // Execute with auto-tuned kernel
    tuner.execute_best(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    d_C.copy_to_host(h_C.data(), M * N * sizeof(float));
    
    float max_error = compare_matrices(h_C.data(), h_C_ref.data(), M * N);
    EXPECT_LT(max_error, 1e-4f) << "Auto-tuned GEMM error: " << max_error;
}

// ============================================================================
// Profiler Tests
// ============================================================================

TEST_F(AdvancedTest, ProfilerBasic) {
    Profiler profiler;
    
    const int M = 512, N = 512, K = 256;
    
    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));
    
    std::vector<float> h_A(M * K), h_B(K * N);
    random_init(h_A.data(), M * K);
    random_init(h_B.data(), K * N);
    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));
    
    auto result = profiler.profile("Optimized GEMM", M, N, K, 5, 20,
        [&]() { launch_optimized_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K); });
    
    EXPECT_GT(result.gflops, 0.0f);
    EXPECT_GT(result.avg_time_ms, 0.0f);
    EXPECT_GT(result.arithmetic_intensity, 0.0f);
    EXPECT_EQ(result.iterations, 20);
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_F(AdvancedTest, StressTestVariousSizes) {
    std::vector<std::tuple<int, int, int>> sizes = {
        {16, 16, 16},
        {33, 65, 17},
        {128, 256, 64},
        {257, 513, 129},
        {512, 512, 512},
        {1000, 1000, 500}
    };
    
    for (const auto& [M, N, K] : sizes) {
        std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_C_ref(M * N);
        random_init(h_A.data(), M * K);
        random_init(h_B.data(), K * N);
        
        cpu_matmul(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
        
        DeviceMemory d_A(M * K * sizeof(float));
        DeviceMemory d_B(K * N * sizeof(float));
        DeviceMemory d_C(M * N * sizeof(float));
        
        d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
        d_B.copy_from_host(h_B.data(), K * N * sizeof(float));
        
        launch_optimized_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        d_C.copy_to_host(h_C.data(), M * N * sizeof(float));
        
        float max_error = compare_matrices(h_C.data(), h_C_ref.data(), M * N);
        EXPECT_LT(max_error, 1e-3f) 
            << "Size " << M << "x" << N << "x" << K << " error: " << max_error;
    }
}
