/**
 * SGEMM Optimization Test Suite
 * 
 * Property-based tests for verifying correctness of all SGEMM kernel implementations.
 * Uses Google Test framework with parameterized tests for comprehensive coverage.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <tuple>
#include <cmath>

#include "utils/cuda_utils.cuh"
#include "utils/verify.cuh"
#include "kernels/naive_sgemm.cuh"
#include "kernels/tiled_sgemm.cuh"
#include "kernels/bank_conflict_free_sgemm.cuh"
#include "kernels/double_buffer_sgemm.cuh"
#include "kernels/tensor_core_sgemm.cuh"

// ============================================================================
// Test Configuration
// ============================================================================

constexpr int PBT_ITERATIONS = 100;
constexpr float STANDARD_RTOL = 1e-4f;
constexpr float STANDARD_ATOL = 1e-5f;
// Tensor Core uses FP16 intermediate precision, which accumulates error
// For 512 K-dimension, expect ~sqrt(512) * FP16_epsilon ≈ 0.01 error
constexpr float TENSOR_CORE_RTOL = 5e-2f;  // 5% relative tolerance
constexpr float TENSOR_CORE_ATOL = 1e-2f;  // 0.01 absolute tolerance

// ============================================================================
// Random Dimension Generator
// ============================================================================

std::vector<std::tuple<int, int, int>> generateRandomDimensions(int count, int alignment = 32) {
    std::vector<std::tuple<int, int, int>> dims;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(1, 32);  // 1-32 tiles
    
    for (int i = 0; i < count; ++i) {
        int M = dist(gen) * alignment;
        int K = dist(gen) * alignment;
        int N = dist(gen) * alignment;
        dims.emplace_back(M, K, N);
    }
    
    return dims;
}

// Standard test dimensions
std::vector<std::tuple<int, int, int>> getStandardDimensions() {
    return {
        {32, 32, 32},
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        // Non-square matrices
        {64, 128, 256},
        {256, 64, 128},
        {128, 256, 64},
        {512, 256, 1024},
    };
}

// ============================================================================
// Property 3: Error Detection Correctness
// Feature: sgemm-optimization, Property 3: Error Detection Correctness
// Validates: Requirements 7.3, 7.4
// ============================================================================

class ErrorDetectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Allocate test matrices
        size_ = 64 * 64;
        h_test_.resize(size_);
        h_ref_.resize(size_);
    }
    
    std::vector<float> h_test_;
    std::vector<float> h_ref_;
    size_t size_;
};

// Test that errors above threshold are correctly flagged for standard kernels
TEST_F(ErrorDetectionTest, StandardKernelErrorDetection) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // Run multiple iterations with random data
    for (int iter = 0; iter < PBT_ITERATIONS; ++iter) {
        // Generate random reference
        for (size_t i = 0; i < size_; ++i) {
            h_ref_[i] = dist(gen);
        }
        
        // Create test data with known error
        float error_magnitude = 1e-3f;  // Above 1e-4 threshold
        for (size_t i = 0; i < size_; ++i) {
            h_test_[i] = h_ref_[i] + error_magnitude * (dist(gen) > 0 ? 1 : -1);
        }
        
        // Verify that error is detected
        VerifyResult result = compareMatrices(h_test_.data(), h_ref_.data(),
                                              64, 64, STANDARD_RTOL, STANDARD_ATOL);
        
        bool should_flag = SGEMMVerifier::shouldFlagAsIncorrect(result.max_rel_error, false);
        EXPECT_TRUE(should_flag) << "Iteration " << iter 
                                  << ": Error above threshold should be flagged";
    }
}

// Test that errors below threshold pass for standard kernels
TEST_F(ErrorDetectionTest, StandardKernelPassesWithinTolerance) {
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int iter = 0; iter < PBT_ITERATIONS; ++iter) {
        // Generate random reference
        for (size_t i = 0; i < size_; ++i) {
            h_ref_[i] = dist(gen);
        }
        
        // Create test data with small error (below threshold)
        float error_magnitude = 1e-6f;  // Well below 1e-4 threshold
        for (size_t i = 0; i < size_; ++i) {
            h_test_[i] = h_ref_[i] + error_magnitude * dist(gen);
        }
        
        VerifyResult result = compareMatrices(h_test_.data(), h_ref_.data(),
                                              64, 64, STANDARD_RTOL, STANDARD_ATOL);
        
        EXPECT_TRUE(result.passed) << "Iteration " << iter 
                                   << ": Error below threshold should pass";
    }
}

// Test Tensor Core threshold (1e-3)
TEST_F(ErrorDetectionTest, TensorCoreErrorDetection) {
    std::mt19937 gen(456);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int iter = 0; iter < PBT_ITERATIONS; ++iter) {
        for (size_t i = 0; i < size_; ++i) {
            h_ref_[i] = dist(gen);
        }
        
        // Error above Tensor Core threshold (1e-3)
        float error_magnitude = 5e-3f;
        for (size_t i = 0; i < size_; ++i) {
            h_test_[i] = h_ref_[i] + error_magnitude * (dist(gen) > 0 ? 1 : -1);
        }
        
        VerifyResult result = compareMatrices(h_test_.data(), h_ref_.data(),
                                              64, 64, TENSOR_CORE_RTOL, TENSOR_CORE_ATOL);
        
        bool should_flag = SGEMMVerifier::shouldFlagAsIncorrect(result.max_rel_error, true);
        EXPECT_TRUE(should_flag) << "Iteration " << iter 
                                  << ": Tensor Core error above threshold should be flagged";
    }
}

// ============================================================================
// Base Test Fixture for Kernel Tests
// ============================================================================

class SGEMMKernelTest : public ::testing::TestWithParam<std::tuple<int, int, int>> {
protected:
    void SetUp() override {
        auto [M, K, N] = GetParam();
        M_ = M;
        K_ = K;
        N_ = N;
        
        // Allocate host memory
        h_A_.resize(M_ * K_);
        h_B_.resize(K_ * N_);
        h_C_.resize(M_ * N_);
        h_C_ref_.resize(M_ * N_);
        
        // Initialize with random values
        initRandomMatrix(h_A_.data(), M_, K_, -1.0f, 1.0f, 42);
        initRandomMatrix(h_B_.data(), K_, N_, -1.0f, 1.0f, 123);
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_A_, M_ * K_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B_, K_ * N_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_, M_ * N_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_ref_, M_ * N_ * sizeof(float)));
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_A_, h_A_.data(), M_ * K_ * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B_, h_B_.data(), K_ * N_ * sizeof(float),
                              cudaMemcpyHostToDevice));
        
        // Compute reference using cuBLAS
        verifier_.computeReference(d_A_, d_B_, d_C_ref_, M_, K_, N_);
        CUDA_CHECK(cudaMemcpy(h_C_ref_.data(), d_C_ref_, M_ * N_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    
    void TearDown() override {
        if (d_A_) cudaFree(d_A_);
        if (d_B_) cudaFree(d_B_);
        if (d_C_) cudaFree(d_C_);
        if (d_C_ref_) cudaFree(d_C_ref_);
    }
    
    int M_, K_, N_;
    std::vector<float> h_A_, h_B_, h_C_, h_C_ref_;
    float *d_A_ = nullptr, *d_B_ = nullptr, *d_C_ = nullptr, *d_C_ref_ = nullptr;
    SGEMMVerifier verifier_;
};

// ============================================================================
// Property 1: Kernel Numerical Correctness (Naive)
// Feature: sgemm-optimization, Property 1: Kernel Numerical Correctness (Naive)
// Validates: Requirements 1.1, 1.3
// ============================================================================

class NaiveSGEMMTest : public SGEMMKernelTest {};

TEST_P(NaiveSGEMMTest, CorrectnessProperty) {
    // Clear output
    CUDA_CHECK(cudaMemset(d_C_, 0, M_ * N_ * sizeof(float)));
    
    // Run naive kernel
    launch_naive_sgemm(d_A_, d_B_, d_C_, M_, K_, N_);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C_.data(), d_C_, M_ * N_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    // Verify
    VerifyResult result = compareMatrices(h_C_.data(), h_C_ref_.data(),
                                          M_, N_, STANDARD_RTOL, STANDARD_ATOL);
    
    EXPECT_TRUE(result.passed) << "Naive SGEMM failed for dimensions "
                               << M_ << "x" << K_ << "x" << N_
                               << " (max_rel_error: " << result.max_rel_error << ")";
}

INSTANTIATE_TEST_SUITE_P(
    StandardDimensions,
    NaiveSGEMMTest,
    ::testing::ValuesIn(getStandardDimensions())
);

// ============================================================================
// Property 1: Kernel Numerical Correctness (Tiled)
// Feature: sgemm-optimization, Property 1: Kernel Numerical Correctness (Tiled)
// Validates: Requirements 2.4
// ============================================================================

class TiledSGEMMTest : public SGEMMKernelTest {};

TEST_P(TiledSGEMMTest, CorrectnessProperty) {
    CUDA_CHECK(cudaMemset(d_C_, 0, M_ * N_ * sizeof(float)));
    
    launch_tiled_sgemm<32>(d_A_, d_B_, d_C_, M_, K_, N_);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_C_.data(), d_C_, M_ * N_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    VerifyResult result = compareMatrices(h_C_.data(), h_C_ref_.data(),
                                          M_, N_, STANDARD_RTOL, STANDARD_ATOL);
    
    EXPECT_TRUE(result.passed) << "Tiled SGEMM failed for dimensions "
                               << M_ << "x" << K_ << "x" << N_
                               << " (max_rel_error: " << result.max_rel_error << ")";
}

INSTANTIATE_TEST_SUITE_P(
    StandardDimensions,
    TiledSGEMMTest,
    ::testing::ValuesIn(getStandardDimensions())
);

// ============================================================================
// Property 1: Kernel Numerical Correctness (BankConflictFree)
// Feature: sgemm-optimization, Property 1: Kernel Numerical Correctness (BankConflictFree)
// Validates: Requirements 3.3
// ============================================================================

class BankConflictFreeSGEMMTest : public SGEMMKernelTest {};

TEST_P(BankConflictFreeSGEMMTest, CorrectnessProperty) {
    CUDA_CHECK(cudaMemset(d_C_, 0, M_ * N_ * sizeof(float)));
    
    launch_bank_conflict_free_sgemm<32>(d_A_, d_B_, d_C_, M_, K_, N_);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_C_.data(), d_C_, M_ * N_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    VerifyResult result = compareMatrices(h_C_.data(), h_C_ref_.data(),
                                          M_, N_, STANDARD_RTOL, STANDARD_ATOL);
    
    EXPECT_TRUE(result.passed) << "BankConflictFree SGEMM failed for dimensions "
                               << M_ << "x" << K_ << "x" << N_
                               << " (max_rel_error: " << result.max_rel_error << ")";
}

INSTANTIATE_TEST_SUITE_P(
    StandardDimensions,
    BankConflictFreeSGEMMTest,
    ::testing::ValuesIn(getStandardDimensions())
);

// ============================================================================
// Property 1: Kernel Numerical Correctness (DoubleBuffer)
// Feature: sgemm-optimization, Property 1: Kernel Numerical Correctness (DoubleBuffer)
// Validates: Requirements 4.4
// ============================================================================

class DoubleBufferSGEMMTest : public SGEMMKernelTest {};

TEST_P(DoubleBufferSGEMMTest, CorrectnessProperty) {
    CUDA_CHECK(cudaMemset(d_C_, 0, M_ * N_ * sizeof(float)));
    
    launch_double_buffer_sgemm<32>(d_A_, d_B_, d_C_, M_, K_, N_);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_C_.data(), d_C_, M_ * N_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    VerifyResult result = compareMatrices(h_C_.data(), h_C_ref_.data(),
                                          M_, N_, STANDARD_RTOL, STANDARD_ATOL);
    
    EXPECT_TRUE(result.passed) << "DoubleBuffer SGEMM failed for dimensions "
                               << M_ << "x" << K_ << "x" << N_
                               << " (max_rel_error: " << result.max_rel_error << ")";
}

INSTANTIATE_TEST_SUITE_P(
    StandardDimensions,
    DoubleBufferSGEMMTest,
    ::testing::ValuesIn(getStandardDimensions())
);

// ============================================================================
// Property 2: Tensor Core Kernel Correctness
// Feature: sgemm-optimization, Property 2: Tensor Core Kernel Correctness
// Validates: Requirements 5.3
// ============================================================================

// Tensor Core dimensions must be multiples of 16
std::vector<std::tuple<int, int, int>> getTensorCoreDimensions() {
    return {
        {16, 16, 16},
        {32, 32, 32},
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        // Non-square (multiples of 16)
        {64, 128, 256},
        {256, 64, 128},
    };
}

class TensorCoreSGEMMTest : public SGEMMKernelTest {};

TEST_P(TensorCoreSGEMMTest, CorrectnessProperty) {
    // Check if GPU supports Tensor Cores (sm_70+)
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (prop.major < 7) {
        GTEST_SKIP() << "Tensor Cores require sm_70 or higher";
    }
    
    CUDA_CHECK(cudaMemset(d_C_, 0, M_ * N_ * sizeof(float)));
    
    launch_tensor_core_sgemm(d_A_, d_B_, d_C_, M_, K_, N_);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_C_.data(), d_C_, M_ * N_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    // Use relaxed tolerance for Tensor Core (FP16 intermediate precision)
    VerifyResult result = compareMatrices(h_C_.data(), h_C_ref_.data(),
                                          M_, N_, TENSOR_CORE_RTOL, TENSOR_CORE_ATOL);
    
    EXPECT_TRUE(result.passed) << "TensorCore SGEMM failed for dimensions "
                               << M_ << "x" << K_ << "x" << N_
                               << " (max_rel_error: " << result.max_rel_error << ")";
}

INSTANTIATE_TEST_SUITE_P(
    TensorCoreDimensions,
    TensorCoreSGEMMTest,
    ::testing::ValuesIn(getTensorCoreDimensions())
);

// ============================================================================
// Property 4: Dimension Invariance
// Feature: sgemm-optimization, Property 4: Dimension Invariance
// Validates: Requirements 1.5, 2.6
// ============================================================================

class DimensionInvarianceTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cublasCreate(&handle_));
    }
    
    void TearDown() override {
        cublasDestroy(handle_);
    }
    
    cublasHandle_t handle_;
};

TEST_F(DimensionInvarianceTest, AllKernelsWorkWithVariousDimensions) {
    std::mt19937 gen(789);
    std::uniform_int_distribution<int> dist(1, 16);  // 1-16 tiles of 32
    
    for (int iter = 0; iter < PBT_ITERATIONS; ++iter) {
        int M = dist(gen) * 32;
        int K = dist(gen) * 32;
        int N = dist(gen) * 32;
        
        // Allocate
        std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_ref(M * N);
        initRandomMatrix(h_A.data(), M, K, -1.0f, 1.0f, iter);
        initRandomMatrix(h_B.data(), K, N, -1.0f, 1.0f, iter + 1000);
        
        float *d_A, *d_B, *d_C, *d_ref;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ref, M * N * sizeof(float)));
        
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
        
        // Compute reference
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, d_B, N, d_A, K, &beta, d_ref, N);
        CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Test each kernel
        auto testKernel = [&](const char* name, auto kernel_func) {
            CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
            kernel_func(d_A, d_B, d_C, M, K, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
            
            VerifyResult result = compareMatrices(h_C.data(), h_ref.data(), M, N,
                                                  STANDARD_RTOL, STANDARD_ATOL);
            EXPECT_TRUE(result.passed) << name << " failed at iteration " << iter
                                       << " with dimensions " << M << "x" << K << "x" << N;
        };
        
        testKernel("Naive", launch_naive_sgemm);
        testKernel("Tiled", launch_tiled_sgemm<32>);
        testKernel("BankConflictFree", launch_bank_conflict_free_sgemm<32>);
        testKernel("DoubleBuffer", launch_double_buffer_sgemm<32>);
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_ref);
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Print GPU info
    printGPUInfo();
    
    return RUN_ALL_TESTS();
}
