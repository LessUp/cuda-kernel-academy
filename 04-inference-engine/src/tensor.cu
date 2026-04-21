#include "tensor.h"
#include "kernels.cuh"

namespace mini_inference {

// ============================================================================
// Tensor Fill Kernel
// ============================================================================

__global__ void fill_kernel(float* data, float value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

void Tensor::fill(float value) {
    if (size_ == 0) return;

    const int block_size = 256;
    const int grid_size = (size_ + block_size - 1) / block_size;
    fill_kernel<<<grid_size, block_size>>>(data_.get(), value, size_);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Element-wise Kernels
// ============================================================================

__global__ void add_kernel(const float* A, const float* B, float* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void add_bias_kernel(const float* X, const float* bias, float* Y,
                                 int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        Y[row * cols + col] = X[row * cols + col] + bias[col];
    }
}

__global__ void relu_kernel(const float* X, float* Y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Y[idx] = fmaxf(X[idx], 0.0f);
    }
}

__global__ void softmax_kernel(const float* X, float* Y, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];

    const float* x_row = X + row * cols;
    float* y_row = Y + row * cols;

    // Find max
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, x_row[i]);
    }

    // Reduce max across threads
    shared[threadIdx.x] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = shared[0];
    __syncthreads();

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float exp_val = expf(x_row[i] - max_val);
        y_row[i] = exp_val;
        sum += exp_val;
    }

    // Reduce sum
    shared[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum = shared[0];
    __syncthreads();

    // Normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        y_row[i] /= sum;
    }
}

// ============================================================================
// Tensor Operations Implementation
// ============================================================================

Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A.ndim() < 2 || B.ndim() < 2) {
        throw std::invalid_argument("matmul requires at least 2D tensors");
    }

    // Get dimensions
    int M = A.dim(A.ndim() - 2);
    int K = A.dim(A.ndim() - 1);
    int N = B.dim(B.ndim() - 1);

    if (K != B.dim(B.ndim() - 2)) {
        throw std::invalid_argument("matmul dimension mismatch");
    }

    // Create output tensor
    std::vector<int> out_shape = A.shape();
    out_shape.back() = N;
    Tensor C(out_shape);

    // Compute batch size
    int batch = 1;
    for (int i = 0; i < A.ndim() - 2; i++) {
        batch *= A.dim(i);
    }

    // Launch kernel for each batch
    for (int b = 0; b < batch; b++) {
        const float* a_ptr = A.data() + b * M * K;
        const float* b_ptr = B.data() + b * K * N;
        float* c_ptr = C.data() + b * M * N;

        launch_optimized_gemm(a_ptr, b_ptr, c_ptr, M, N, K);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return C;
}

Tensor add(const Tensor& A, const Tensor& B) {
    if (A.size() != B.size()) {
        throw std::invalid_argument("add: tensor sizes must match");
    }

    Tensor C(A.shape());

    const int block_size = 256;
    const int grid_size = (A.size() + block_size - 1) / block_size;
    add_kernel<<<grid_size, block_size>>>(A.data(), B.data(), C.data(), A.size());
    CUDA_CHECK(cudaGetLastError());

    return C;
}

Tensor add_bias(const Tensor& X, const Tensor& bias) {
    if (X.ndim() < 1 || bias.ndim() != 1) {
        throw std::invalid_argument("add_bias: invalid dimensions");
    }

    int cols = X.dim(X.ndim() - 1);
    if (cols != bias.dim(0)) {
        throw std::invalid_argument("add_bias: bias size mismatch");
    }

    int rows = X.size() / cols;
    Tensor Y(X.shape());

    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    add_bias_kernel<<<grid, block>>>(X.data(), bias.data(), Y.data(), rows, cols);
    CUDA_CHECK(cudaGetLastError());

    return Y;
}

Tensor relu(const Tensor& X) {
    Tensor Y(X.shape());

    const int block_size = 256;
    const int grid_size = (X.size() + block_size - 1) / block_size;
    relu_kernel<<<grid_size, block_size>>>(X.data(), Y.data(), X.size());
    CUDA_CHECK(cudaGetLastError());

    return Y;
}

Tensor softmax(const Tensor& X) {
    if (X.ndim() < 1) {
        throw std::invalid_argument("softmax: tensor must be at least 1D");
    }

    int cols = X.dim(X.ndim() - 1);
    int rows = X.size() / cols;

    Tensor Y(X.shape());

    int block_size = 1;
    while (block_size < cols && block_size < 256) {
        block_size <<= 1;
    }
    softmax_kernel<<<rows, block_size, block_size * sizeof(float)>>>(
        X.data(), Y.data(), rows, cols);
    CUDA_CHECK(cudaGetLastError());

    return Y;
}

Tensor fused_linear_relu(const Tensor& X, const Tensor& W, const Tensor& bias) {
    if (X.ndim() < 2 || W.ndim() != 2 || bias.ndim() != 1) {
        throw std::invalid_argument("fused_linear_relu: invalid dimensions");
    }

    int M = X.dim(X.ndim() - 2);
    int K = X.dim(X.ndim() - 1);
    int N = W.dim(1);

    if (K != W.dim(0) || N != bias.dim(0)) {
        throw std::invalid_argument("fused_linear_relu: dimension mismatch");
    }

    std::vector<int> out_shape = X.shape();
    out_shape.back() = N;
    Tensor Y(out_shape);

    int batch = 1;
    for (int i = 0; i < X.ndim() - 2; i++) {
        batch *= X.dim(i);
    }

    for (int b = 0; b < batch; b++) {
        const float* x_ptr = X.data() + b * M * K;
        float* y_ptr = Y.data() + b * M * N;

        launch_fused_gemm(x_ptr, W.data(), y_ptr, bias.data(),
                          M, N, K, true, true);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return Y;
}

} // namespace mini_inference
