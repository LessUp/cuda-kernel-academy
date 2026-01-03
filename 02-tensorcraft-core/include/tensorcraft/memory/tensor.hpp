#pragma once
/**
 * @file tensor.hpp
 * @brief GPU Tensor wrapper with RAII memory management
 * 
 * Provides a type-safe tensor class that automatically manages
 * GPU memory allocation and deallocation using RAII.
 */

#include "../core/cuda_check.hpp"
#include "../core/type_traits.hpp"
#include <vector>
#include <numeric>
#include <memory>
#include <cassert>

namespace tensorcraft {

/**
 * @brief GPU Tensor with RAII memory management
 * 
 * @tparam T Element type (float, half, etc.)
 * 
 * Features:
 * - Automatic GPU memory allocation/deallocation
 * - Move semantics (no copy to prevent accidental copies)
 * - Host-device data transfer utilities
 * - Shape and stride information
 */
template<typename T>
class Tensor {
public:
    using value_type = T;
    using size_type = size_t;
    using shape_type = std::vector<size_t>;
    
    /// Default constructor (empty tensor)
    Tensor() = default;
    
    /**
     * @brief Construct tensor with given shape
     * @param shape Dimensions of the tensor
     */
    explicit Tensor(const shape_type& shape) 
        : shape_(shape)
        , size_(compute_size(shape))
        , strides_(compute_strides(shape)) {
        if (size_ > 0) {
            TC_CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(T)));
        }
    }
    
    /**
     * @brief Construct tensor with initializer list shape
     */
    Tensor(std::initializer_list<size_t> shape)
        : Tensor(shape_type(shape)) {}
    
    /**
     * @brief Construct 1D tensor
     */
    explicit Tensor(size_t n) : Tensor(shape_type{n}) {}
    
    /**
     * @brief Construct 2D tensor
     */
    Tensor(size_t rows, size_t cols) : Tensor(shape_type{rows, cols}) {}
    
    /**
     * @brief Construct 3D tensor
     */
    Tensor(size_t d0, size_t d1, size_t d2) : Tensor(shape_type{d0, d1, d2}) {}
    
    /**
     * @brief Construct 4D tensor
     */
    Tensor(size_t d0, size_t d1, size_t d2, size_t d3) 
        : Tensor(shape_type{d0, d1, d2, d3}) {}
    
    /// Destructor - frees GPU memory
    ~Tensor() {
        if (data_) {
            cudaFree(data_);
            data_ = nullptr;
        }
    }
    
    // Move constructor
    Tensor(Tensor&& other) noexcept
        : data_(other.data_)
        , shape_(std::move(other.shape_))
        , strides_(std::move(other.strides_))
        , size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (data_) {
                cudaFree(data_);
            }
            data_ = other.data_;
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Disable copy (use clone() for explicit copy)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    /// Get raw data pointer
    T* data() { return data_; }
    const T* data() const { return data_; }
    
    /// Get total number of elements
    size_type size() const { return size_; }
    
    /// Get size in bytes
    size_type bytes() const { return size_ * sizeof(T); }
    
    /// Get shape
    const shape_type& shape() const { return shape_; }
    
    /// Get strides
    const shape_type& strides() const { return strides_; }
    
    /// Get number of dimensions
    size_t ndim() const { return shape_.size(); }
    
    /// Get dimension size
    size_t dim(size_t i) const { 
        assert(i < shape_.size());
        return shape_[i]; 
    }
    
    /// Check if tensor is empty
    bool empty() const { return size_ == 0; }
    
    /// Check if tensor is contiguous
    bool is_contiguous() const {
        if (shape_.empty()) return true;
        size_t expected_stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
            if (strides_[i] != expected_stride) return false;
            expected_stride *= shape_[i];
        }
        return true;
    }
    
    // ========================================================================
    // Data Transfer
    // ========================================================================
    
    /**
     * @brief Copy data from host to device
     */
    void copy_from_host(const T* host_data) {
        assert(data_ && host_data);
        TC_CUDA_CHECK(cudaMemcpy(data_, host_data, bytes(), cudaMemcpyHostToDevice));
    }
    
    /**
     * @brief Copy data from host vector
     */
    void copy_from_host(const std::vector<T>& host_data) {
        assert(host_data.size() == size_);
        copy_from_host(host_data.data());
    }
    
    /**
     * @brief Copy data from device to host
     */
    void copy_to_host(T* host_data) const {
        assert(data_ && host_data);
        TC_CUDA_CHECK(cudaMemcpy(host_data, data_, bytes(), cudaMemcpyDeviceToHost));
    }
    
    /**
     * @brief Copy data to host vector
     */
    std::vector<T> to_host() const {
        std::vector<T> result(size_);
        copy_to_host(result.data());
        return result;
    }
    
    /**
     * @brief Async copy from host
     */
    void copy_from_host_async(const T* host_data, cudaStream_t stream) {
        assert(data_ && host_data);
        TC_CUDA_CHECK(cudaMemcpyAsync(data_, host_data, bytes(), 
                                       cudaMemcpyHostToDevice, stream));
    }
    
    /**
     * @brief Async copy to host
     */
    void copy_to_host_async(T* host_data, cudaStream_t stream) const {
        assert(data_ && host_data);
        TC_CUDA_CHECK(cudaMemcpyAsync(host_data, data_, bytes(), 
                                       cudaMemcpyDeviceToHost, stream));
    }
    
    // ========================================================================
    // Memory Operations
    // ========================================================================
    
    /**
     * @brief Fill tensor with value
     */
    void fill(T value) {
        if constexpr (sizeof(T) == 1) {
            TC_CUDA_CHECK(cudaMemset(data_, static_cast<int>(value), bytes()));
        } else {
            // For larger types, need to use a kernel
            std::vector<T> host_data(size_, value);
            copy_from_host(host_data);
        }
    }
    
    /**
     * @brief Zero the tensor
     */
    void zero() {
        TC_CUDA_CHECK(cudaMemset(data_, 0, bytes()));
    }
    
    /**
     * @brief Create a deep copy
     */
    Tensor clone() const {
        Tensor result(shape_);
        if (size_ > 0) {
            TC_CUDA_CHECK(cudaMemcpy(result.data_, data_, bytes(), 
                                      cudaMemcpyDeviceToDevice));
        }
        return result;
    }
    
    /**
     * @brief Reshape tensor (must have same total size)
     */
    Tensor& reshape(const shape_type& new_shape) {
        size_t new_size = compute_size(new_shape);
        assert(new_size == size_ && "Reshape must preserve total size");
        shape_ = new_shape;
        strides_ = compute_strides(new_shape);
        return *this;
    }
    
    // ========================================================================
    // Static Factory Methods
    // ========================================================================
    
    /**
     * @brief Create tensor from host data
     */
    static Tensor from_host(const T* data, const shape_type& shape) {
        Tensor result(shape);
        result.copy_from_host(data);
        return result;
    }
    
    /**
     * @brief Create tensor from host vector
     */
    static Tensor from_host(const std::vector<T>& data, const shape_type& shape) {
        assert(data.size() == compute_size(shape));
        return from_host(data.data(), shape);
    }
    
    /**
     * @brief Create zero-initialized tensor
     */
    static Tensor zeros(const shape_type& shape) {
        Tensor result(shape);
        result.zero();
        return result;
    }
    
    /**
     * @brief Create tensor filled with ones
     */
    static Tensor ones(const shape_type& shape) {
        Tensor result(shape);
        result.fill(T(1));
        return result;
    }

private:
    static size_type compute_size(const shape_type& shape) {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), 
                               size_type(1), std::multiplies<size_type>());
    }
    
    static shape_type compute_strides(const shape_type& shape) {
        if (shape.empty()) return {};
        shape_type strides(shape.size());
        size_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }
    
    T* data_ = nullptr;
    shape_type shape_;
    shape_type strides_;
    size_type size_ = 0;
};

// Type aliases for common tensor types
using FloatTensor = Tensor<float>;
using HalfTensor = Tensor<__half>;
using BFloat16Tensor = Tensor<__nv_bfloat16>;
using IntTensor = Tensor<int>;
using Int8Tensor = Tensor<int8_t>;

} // namespace tensorcraft
