#pragma once

#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

#include "common.h"
#include "memory_pool.h"

namespace mini_inference {

// ============================================================================
// Tensor Class
// N-dimensional tensor with GPU storage
// ============================================================================

class Tensor {
public:
    Tensor() = default;

    // Create tensor with shape
    explicit Tensor(const std::vector<int>& shape) : shape_(shape) {
        size_ = compute_size();
        if (size_ > 0) {
            data_.allocate(size_ * sizeof(float));
        }
        compute_strides();
    }

    // Create tensor with shape and data
    Tensor(const std::vector<int>& shape, const float* data) : shape_(shape) {
        size_ = compute_size();
        if (size_ > 0) {
            data_.allocate(size_ * sizeof(float));
            data_.copy_from_host(data, size_ * sizeof(float));
        }
        compute_strides();
    }

    // Move operations
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    // Disable copy (expensive for GPU data)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Clone tensor
    Tensor clone() const {
        Tensor result(shape_);
        if (size_ > 0) {
            CUDA_CHECK(cudaMemcpy(result.data_.get(), data_.get(), size_ * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }
        return result;
    }

    // Reshape (must have same total size)
    void reshape(const std::vector<int>& new_shape) {
        size_t new_size = 1;
        for (int dim : new_shape) {
            new_size *= dim;
        }
        if (new_size != size_) {
            throw std::invalid_argument("Reshape size mismatch");
        }
        shape_ = new_shape;
        compute_strides();
    }

    // View as 2D matrix
    MatrixDesc as_matrix() const {
        if (shape_.size() < 2) {
            throw std::invalid_argument("Tensor must be at least 2D for matrix view");
        }
        int rows = 1;
        for (size_t i = 0; i < shape_.size() - 1; i++) {
            rows *= shape_[i];
        }
        int cols = shape_.back();
        return MatrixDesc(const_cast<float*>(data_.get()), rows, cols, cols);
    }

    // Accessors
    const std::vector<int>& shape() const { return shape_; }
    const std::vector<int>& strides() const { return strides_; }
    size_t size() const { return size_; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    int dim(int i) const { return shape_[i]; }

    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }

    bool empty() const { return size_ == 0; }

    // Copy to/from host
    void copy_from_host(const float* host_data) {
        data_.copy_from_host(host_data, size_ * sizeof(float));
    }

    void copy_to_host(float* host_data) const {
        data_.copy_to_host(host_data, size_ * sizeof(float));
    }

    std::vector<float> to_host() const {
        std::vector<float> result(size_);
        copy_to_host(result.data());
        return result;
    }

    // Fill with value
    void fill(float value);

    // Zero tensor
    void zero() { data_.zero(); }

    // Print info
    void print_info() const {
        std::cout << "Tensor(shape=[";
        for (size_t i = 0; i < shape_.size(); i++) {
            std::cout << shape_[i];
            if (i < shape_.size() - 1)
                std::cout << ", ";
        }
        std::cout << "], size=" << size_ << ", dtype=float32)" << std::endl;
    }

private:
    size_t compute_size() const {
        if (shape_.empty())
            return 0;
        return std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    }

    void compute_strides() {
        strides_.resize(shape_.size());
        if (shape_.empty())
            return;

        strides_.back() = 1;
        for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; i--) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

    std::vector<int> shape_;
    std::vector<int> strides_;
    size_t size_ = 0;
    PooledMemory data_;
};

// ============================================================================
// Tensor Operations
// ============================================================================

// Matrix multiplication: C = A @ B
Tensor matmul(const Tensor& A, const Tensor& B);

// Element-wise addition: C = A + B
Tensor add(const Tensor& A, const Tensor& B);

// Add bias: Y = X + bias (broadcast along last dimension)
Tensor add_bias(const Tensor& X, const Tensor& bias);

// ReLU activation
Tensor relu(const Tensor& X);

// Softmax along last dimension
Tensor softmax(const Tensor& X);

// Fused matmul + bias + relu
Tensor fused_linear_relu(const Tensor& X, const Tensor& W, const Tensor& bias);

}  // namespace mini_inference
