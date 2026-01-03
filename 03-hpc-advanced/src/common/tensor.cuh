#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <concepts>
#include <cstring>
#include "cuda_check.cuh"

namespace hpc {

// Concept for CUDA-compatible numeric types
template <typename T>
concept CudaNumeric = std::is_arithmetic_v<T> ||
                      std::is_same_v<T, __half> ||
                      std::is_same_v<T, __nv_bfloat16>;

// RAII wrapper for GPU memory
template <CudaNumeric T>
class Tensor {
public:
    explicit Tensor(size_t size) : size_(size), data_(nullptr) {
        if (size_ > 0) {
            CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(T)));
        }
    }

    ~Tensor() {
        if (data_) {
            cudaFree(data_);
        }
    }

    // Move semantics
    Tensor(Tensor&& other) noexcept
        : size_(other.size_), data_(other.data_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (data_) cudaFree(data_);
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Delete copy operations
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Accessors
    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] size_t bytes() const noexcept { return size_ * sizeof(T); }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    // Host-device transfers
    void copy_from_host(const T* host_data) {
        CUDA_CHECK(cudaMemcpy(data_, host_data, bytes(), cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* host_data) const {
        CUDA_CHECK(cudaMemcpy(host_data, data_, bytes(), cudaMemcpyDeviceToHost));
    }

    void copy_from_host(const std::vector<T>& host_vec) {
        copy_from_host(host_vec.data());
    }

    [[nodiscard]] std::vector<T> to_host() const {
        std::vector<T> result(size_);
        copy_to_host(result.data());
        return result;
    }

    // Fill with zeros
    void zero() {
        CUDA_CHECK(cudaMemset(data_, 0, bytes()));
    }

    // Async versions
    void copy_from_host_async(const T* host_data, cudaStream_t stream) {
        CUDA_CHECK(cudaMemcpyAsync(data_, host_data, bytes(),
                                   cudaMemcpyHostToDevice, stream));
    }

    void copy_to_host_async(T* host_data, cudaStream_t stream) const {
        CUDA_CHECK(cudaMemcpyAsync(host_data, data_, bytes(),
                                   cudaMemcpyDeviceToHost, stream));
    }

private:
    size_t size_;
    T* data_;
};

} // namespace hpc
