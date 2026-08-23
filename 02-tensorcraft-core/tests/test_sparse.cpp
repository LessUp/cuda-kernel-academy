/**
 * @file test_sparse.cpp
 * @brief Basic correctness tests for CSR sparse operations
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "tensorcraft/core/cuda_check.hpp"
#include "tensorcraft/kernels/sparse.hpp"

using namespace tensorcraft::kernels;

namespace {

// Reference conversions / ops on CPU.
struct CpuCsr {
    std::vector<float> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptrs;
    int rows = 0, cols = 0, nnz = 0;
};

CpuCsr dense_to_csr_cpu(const std::vector<float>& dense, int rows, int cols) {
    CpuCsr csr;
    csr.rows = rows;
    csr.cols = cols;
    csr.row_ptrs.assign(rows + 1, 0);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (std::fabs(dense[r * cols + c]) > 0.0f) {
                csr.values.push_back(dense[r * cols + c]);
                csr.col_indices.push_back(c);
            }
        }
        csr.row_ptrs[r + 1] = static_cast<int>(csr.values.size());
    }
    csr.nnz = static_cast<int>(csr.values.size());
    return csr;
}

std::vector<float> spmv_cpu(const CpuCsr& A, const std::vector<float>& x) {
    std::vector<float> y(A.rows, 0.0f);
    for (int r = 0; r < A.rows; ++r) {
        float sum = 0.0f;
        for (int i = A.row_ptrs[r]; i < A.row_ptrs[r + 1]; ++i) {
            sum += A.values[i] * x[A.col_indices[i]];
        }
        y[r] = sum;
    }
    return y;
}

std::vector<float> spmm_cpu(const CpuCsr& A, const std::vector<float>& B, int N) {
    std::vector<float> C(A.rows * N, 0.0f);
    for (int r = 0; r < A.rows; ++r) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int i = A.row_ptrs[r]; i < A.row_ptrs[r + 1]; ++i) {
                sum += A.values[i] * B[A.col_indices[i] * N + n];
            }
            C[r * N + n] = sum;
        }
    }
    return C;
}

}  // namespace

class SparseTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices found.";
        }
        TC_CUDA_CHECK(cudaSetDevice(0));
    }

    // Sparse matrix with ~50% exact zeros.
    std::vector<float> random_sparse(size_t n) {
        std::mt19937 gen(7);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> v(n);
        for (size_t i = 0; i < n; ++i) {
            v[i] = (i % 2 == 0) ? dist(gen) : 0.0f;
        }
        return v;
    }

    std::vector<float> random_dense(size_t n) {
        std::mt19937 gen(11);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> v(n);
        for (auto& x : v)
            x = dist(gen);
        return v;
    }

    // GPU dense -> CSR using the library kernels (row counts + prefix sum on host).
    void dense_to_csr_gpu(const std::vector<float>& dense, int rows, int cols, float*& d_values,
                          int*& d_col_indices, int*& d_row_ptrs, int& nnz) {
        float* d_dense = nullptr;
        int* d_row_counts = nullptr;
        TC_CUDA_CHECK(cudaMalloc(&d_dense, dense.size() * sizeof(float)));
        TC_CUDA_CHECK(cudaMalloc(&d_row_counts, rows * sizeof(int)));
        TC_CUDA_CHECK(cudaMemcpy(d_dense, dense.data(), dense.size() * sizeof(float),
                                 cudaMemcpyHostToDevice));

        count_nnz_per_row_kernel<float>
            <<<(rows + 255) / 256, 256>>>(d_dense, d_row_counts, rows, cols, 0.0f);
        TC_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> row_counts(rows);
        TC_CUDA_CHECK(cudaMemcpy(row_counts.data(), d_row_counts, rows * sizeof(int),
                                 cudaMemcpyDeviceToHost));

        std::vector<int> row_ptrs(rows + 1, 0);
        for (int r = 0; r < rows; ++r) {
            row_ptrs[r + 1] = row_ptrs[r] + row_counts[r];
        }
        nnz = row_ptrs[rows];

        TC_CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));
        TC_CUDA_CHECK(cudaMalloc(&d_col_indices, nnz * sizeof(int)));
        TC_CUDA_CHECK(cudaMalloc(&d_row_ptrs, (rows + 1) * sizeof(int)));
        TC_CUDA_CHECK(cudaMemcpy(d_row_ptrs, row_ptrs.data(), (rows + 1) * sizeof(int),
                                 cudaMemcpyHostToDevice));

        dense_to_csr_kernel<float><<<(rows + 255) / 256, 256>>>(d_dense, d_values, d_col_indices,
                                                                d_row_ptrs, rows, cols, 0.0f);
        TC_CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_dense);
        cudaFree(d_row_counts);
    }
};

// dense -> CSR -> dense round trip.
TEST_F(SparseTest, DenseCsrRoundTrip) {
    const int rows = 16, cols = 12;
    auto dense = random_sparse(rows * cols);

    float* d_values = nullptr;
    int* d_col_indices = nullptr;
    int* d_row_ptrs = nullptr;
    int nnz = 0;
    dense_to_csr_gpu(dense, rows, cols, d_values, d_col_indices, d_row_ptrs, nnz);

    // Round trip back to dense.
    std::vector<float> round_trip(rows * cols, -999.0f);
    float* d_dense_out = nullptr;
    TC_CUDA_CHECK(cudaMalloc(&d_dense_out, rows * cols * sizeof(float)));
    launch_csr_to_dense<float>(d_values, d_col_indices, d_row_ptrs, d_dense_out, rows, cols);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
    TC_CUDA_CHECK(cudaMemcpy(round_trip.data(), d_dense_out, rows * cols * sizeof(float),
                             cudaMemcpyDeviceToHost));

    for (int i = 0; i < rows * cols; ++i) {
        EXPECT_FLOAT_EQ(round_trip[i], dense[i]) << "element " << i;
    }

    cudaFree(d_values);
    cudaFree(d_col_indices);
    cudaFree(d_row_ptrs);
    cudaFree(d_dense_out);
}

// SpMV (CSR) vs CPU reference, both vectorized and scalar variants.
TEST_F(SparseTest, SpmvCsrMatchesCpu) {
    const int rows = 32, cols = 16;
    auto dense = random_sparse(rows * cols);
    auto x = random_dense(cols);

    float* d_values = nullptr;
    int* d_col_indices = nullptr;
    int* d_row_ptrs = nullptr;
    int nnz = 0;
    dense_to_csr_gpu(dense, rows, cols, d_values, d_col_indices, d_row_ptrs, nnz);

    auto A_cpu = dense_to_csr_cpu(dense, rows, cols);
    auto expected = spmv_cpu(A_cpu, x);

    float *d_x, *d_y;
    TC_CUDA_CHECK(cudaMalloc(&d_x, cols * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(float)));
    TC_CUDA_CHECK(cudaMemcpy(d_x, x.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> actual(rows);
    for (bool use_vector : {true, false}) {
        launch_spmv_csr<float>(d_values, d_col_indices, d_row_ptrs, d_x, d_y, rows, use_vector);
        TC_CUDA_CHECK(cudaDeviceSynchronize());
        TC_CUDA_CHECK(cudaMemcpy(actual.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));
        for (int r = 0; r < rows; ++r) {
            EXPECT_NEAR(actual[r], expected[r], 1e-4f) << "row " << r << " vector=" << use_vector;
        }
    }

    cudaFree(d_values);
    cudaFree(d_col_indices);
    cudaFree(d_row_ptrs);
    cudaFree(d_x);
    cudaFree(d_y);
}

// SpMM (CSR * dense) vs CPU reference.
TEST_F(SparseTest, SpmmCsrMatchesCpu) {
    const int rows = 16, K = 12, N = 8;
    auto dense = random_sparse(rows * K);
    auto B = random_dense(K * N);

    float* d_values = nullptr;
    int* d_col_indices = nullptr;
    int* d_row_ptrs = nullptr;
    int nnz = 0;
    dense_to_csr_gpu(dense, rows, K, d_values, d_col_indices, d_row_ptrs, nnz);

    auto A_cpu = dense_to_csr_cpu(dense, rows, K);
    auto expected = spmm_cpu(A_cpu, B, N);

    float *d_B, *d_C;
    TC_CUDA_CHECK(cudaMalloc(&d_B, B.size() * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_C, rows * N * sizeof(float)));
    TC_CUDA_CHECK(cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch_spmm_csr<float>(d_values, d_col_indices, d_row_ptrs, d_B, d_C, rows, K, N);
    TC_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual(rows * N);
    TC_CUDA_CHECK(cudaMemcpy(actual.data(), d_C, rows * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < rows * N; ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-4f) << "element " << i;
    }

    cudaFree(d_values);
    cudaFree(d_col_indices);
    cudaFree(d_row_ptrs);
    cudaFree(d_B);
    cudaFree(d_C);
}
