#pragma once
/**
 * @file device_memory.hpp
 * @brief RAII wrapper for CUDA device memory
 *
 * Provides a templated RAII wrapper for managing device memory allocations.
 * Supports automatic cleanup, move semantics, and host-device data transfer.
 *
 * @example
 * @code
 * #include <cuda_academy/core/device_memory.hpp>
 *
 * // Allocate device memory
 * cuda_academy::DeviceMemory<float> d_data(1024);
 *
 * // Copy from host
 * std::vector<float> host_data(1024, 1.0f);
 * d_data.copy_from_host(host_data);
 *
 * // Use in kernel
 * my_kernel<<<grid, block>>>(d_data.get(), 1024);
 *
 * // Copy back to host
 * d_data.copy_to_host(host_data);
 * @endcode
 */

#include <cuda_academy/core/cuda_check.hpp>

#include <vector>

namespace cuda_academy {

// ============================================================================
// DeviceMemory - RAII Wrapper for Device Memory
// ============================================================================

/**
 * @brief RAII wrapper for CUDA device memory
 * @tparam T Element type (must be trivially copyable for safe device transfer)
 *
 * Manages a contiguous block of device memory with automatic cleanup.
 * Supports move semantics but not copy (to prevent accidental duplication).
 */
template <typename T>
class DeviceMemory {
public:
    /// Create an empty (null) allocation
    DeviceMemory() : ptr_(nullptr), size_(0) {}

    /**
     * @brief Allocate device memory for count elements
     * @param count Number of elements to allocate
     * @throws CudaError if allocation fails
     */
    explicit DeviceMemory(size_t count) : ptr_(nullptr), size_(count) {
        if (count > 0) {
            CA_CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        }
    }

    /// Free device memory
    ~DeviceMemory() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    // ------------------------------------------------------------------------
    // Move Semantics
    // ------------------------------------------------------------------------

    DeviceMemory(DeviceMemory&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Disable copy
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    // ------------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------------

    /// Get raw device pointer
    [[nodiscard]] T* get() noexcept { return ptr_; }

    /// Get const raw device pointer
    [[nodiscard]] const T* get() const noexcept { return ptr_; }

    /// Get number of elements
    [[nodiscard]] size_t size() const noexcept { return size_; }

    /// Get size in bytes
    [[nodiscard]] size_t bytes() const noexcept { return size_ * sizeof(T); }

    /// Check if allocation is empty
    [[nodiscard]] bool empty() const noexcept { return ptr_ == nullptr || size_ == 0; }

    // ------------------------------------------------------------------------
    // Host-Device Transfers
    // ------------------------------------------------------------------------

    /**
     * @brief Copy all data from host to device
     * @param host_ptr Pointer to host data (must have at least size_ elements)
     * @throws CudaError if copy fails
     */
    void copy_from_host(const T* host_ptr) {
        CA_CUDA_CHECK(cudaMemcpy(ptr_, host_ptr, size_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    /**
     * @brief Copy partial data from host to device
     * @param host_ptr Pointer to host data
     * @param count Number of elements to copy
     * @throws CudaError if copy fails
     */
    void copy_from_host(const T* host_ptr, size_t count) {
        CA_CUDA_CHECK(cudaMemcpy(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    /**
     * @brief Copy data from host vector to device
     * @param host_vec Host vector
     * @throws CudaError if copy fails
     */
    void copy_from_host(const std::vector<T>& host_vec) {
        copy_from_host(host_vec.data(), host_vec.size());
    }

    /**
     * @brief Copy all data from device to host
     * @param host_ptr Pointer to host destination (must have space for size_ elements)
     * @throws CudaError if copy fails
     */
    void copy_to_host(T* host_ptr) const {
        CA_CUDA_CHECK(cudaMemcpy(host_ptr, ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    /**
     * @brief Copy partial data from device to host
     * @param host_ptr Pointer to host destination
     * @param count Number of elements to copy
     * @throws CudaError if copy fails
     */
    void copy_to_host(T* host_ptr, size_t count) const {
        CA_CUDA_CHECK(cudaMemcpy(host_ptr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    /**
     * @brief Copy data to host vector
     * @return Host vector containing device data
     * @throws CudaError if copy fails
     */
    [[nodiscard]] std::vector<T> to_host() const {
        std::vector<T> result(size_);
        copy_to_host(result.data());
        return result;
    }

    // ------------------------------------------------------------------------
    // Async Host-Device Transfers
    // ------------------------------------------------------------------------

    /**
     * @brief Asynchronously copy all data from host to device
     * @param host_ptr Pointer to host data
     * @param stream CUDA stream
     * @throws CudaError if copy fails
     */
    void copy_from_host_async(const T* host_ptr, cudaStream_t stream) {
        CA_CUDA_CHECK(
            cudaMemcpyAsync(ptr_, host_ptr, size_ * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    /**
     * @brief Asynchronously copy partial data from host to device
     * @param host_ptr Pointer to host data
     * @param stream CUDA stream
     * @param count Number of elements to copy
     * @throws CudaError if copy fails
     */
    void copy_from_host_async(const T* host_ptr, cudaStream_t stream, size_t count) {
        CA_CUDA_CHECK(
            cudaMemcpyAsync(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    /**
     * @brief Asynchronously copy all data from device to host
     * @param host_ptr Pointer to host destination
     * @param stream CUDA stream
     * @throws CudaError if copy fails
     */
    void copy_to_host_async(T* host_ptr, cudaStream_t stream) const {
        CA_CUDA_CHECK(
            cudaMemcpyAsync(host_ptr, ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

    /**
     * @brief Asynchronously copy partial data from device to host
     * @param host_ptr Pointer to host destination
     * @param stream CUDA stream
     * @param count Number of elements to copy
     * @throws CudaError if copy fails
     */
    void copy_to_host_async(T* host_ptr, cudaStream_t stream, size_t count) const {
        CA_CUDA_CHECK(
            cudaMemcpyAsync(host_ptr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

    // ------------------------------------------------------------------------
    // Memory Operations
    // ------------------------------------------------------------------------

    /**
     * @brief Fill device memory with zeros
     * @throws CudaError if memset fails
     */
    void zero() {
        if (ptr_ && size_ > 0) {
            CA_CUDA_CHECK(cudaMemset(ptr_, 0, size_ * sizeof(T)));
        }
    }

    /**
     * @brief Reallocate device memory
     * @param count New number of elements
     * @throws CudaError if allocation fails
     *
     * Note: On allocation failure, the object remains in its original state
     * (old allocation preserved).
     */
    void resize(size_t count) {
        // Allocate new memory first (before freeing old)
        T* new_ptr = nullptr;
        if (count > 0) {
            CA_CUDA_CHECK(cudaMalloc(&new_ptr, count * sizeof(T)));
        }

        // Only free old memory after successful allocation
        if (ptr_) {
            cudaFree(ptr_);
        }

        ptr_ = new_ptr;
        size_ = count;
    }

private:
    T* ptr_;
    size_t size_;
};

// ============================================================================
// Type Aliases for Common Cases
// ============================================================================

using DeviceMemoryF = DeviceMemory<float>;
using DeviceMemoryD = DeviceMemory<double>;
using DeviceMemoryI = DeviceMemory<int>;

}  // namespace cuda_academy
