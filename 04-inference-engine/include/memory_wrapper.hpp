#pragma once

/**
 * @file memory_wrapper.hpp
 * @brief Generic GPU memory wrapper supporting different allocators
 *
 * Provides a RAII wrapper for GPU memory that can work with any
 * MemoryAllocator implementation, enabling flexible memory management.
 */

#include "common.h"
#include "memory_allocator.hpp"
#include "memory_pool.h"

namespace mini_inference {

/**
 * @brief Generic RAII wrapper for GPU memory
 *
 * Uses any MemoryAllocator for allocation, defaulting to MemoryPool
 * for efficient pooled allocation. Can be configured to use other
 * allocators for testing or special use cases.
 */
class MemoryWrapper {
public:
    MemoryWrapper() = default;

    explicit MemoryWrapper(size_t bytes, MemoryAllocator* allocator = nullptr)
        : allocator_(allocator ? allocator : &MemoryPool::instance()) {
        if (bytes > 0) {
            ptr_ = static_cast<float*>(allocator_->allocate(bytes));
            size_ = bytes;
        }
    }

    ~MemoryWrapper() {
        if (ptr_ && allocator_) {
            allocator_->deallocate(ptr_);
        }
    }

    // Disable copy
    MemoryWrapper(const MemoryWrapper&) = delete;
    MemoryWrapper& operator=(const MemoryWrapper&) = delete;

    // Enable move
    MemoryWrapper(MemoryWrapper&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_), allocator_(other.allocator_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    MemoryWrapper& operator=(MemoryWrapper&& other) noexcept {
        if (this != &other) {
            if (ptr_ && allocator_) {
                allocator_->deallocate(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            allocator_ = other.allocator_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void allocate(size_t bytes, MemoryAllocator* allocator = nullptr) {
        if (ptr_ && allocator_) {
            allocator_->deallocate(ptr_);
        }
        ptr_ = nullptr;
        size_ = 0;

        if (allocator) {
            allocator_ = allocator;
        } else if (!allocator_) {
            // Default-constructed wrapper must still allocate from the
            // process-wide pool; otherwise Tensor({shape}) leaves ptr_ null
            // and later copy/fill operations fail with cudaMemcpy errors.
            allocator_ = &MemoryPool::instance();
        }

        if (bytes > 0 && allocator_) {
            ptr_ = static_cast<float*>(allocator_->allocate(bytes));
            size_ = bytes;
        }
    }

    void free() {
        if (ptr_ && allocator_) {
            allocator_->deallocate(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    float* get() { return ptr_; }
    const float* get() const { return ptr_; }
    size_t size() const { return size_; }
    bool empty() const { return ptr_ == nullptr; }
    MemoryAllocator* allocator() const { return allocator_; }

    void copy_from_host(const float* host_data, size_t bytes) {
        CUDA_CHECK(cudaMemcpy(ptr_, host_data, bytes, cudaMemcpyHostToDevice));
    }

    void copy_to_host(float* host_data, size_t bytes) const {
        CUDA_CHECK(cudaMemcpy(host_data, ptr_, bytes, cudaMemcpyDeviceToHost));
    }

    void zero() {
        if (ptr_ && size_ > 0) {
            CUDA_CHECK(cudaMemset(ptr_, 0, size_));
        }
    }

private:
    float* ptr_ = nullptr;
    size_t size_ = 0;
    MemoryAllocator* allocator_ = nullptr;
};

// Keep PooledMemory as an alias for backward compatibility
// Note: PooledMemory is still defined in memory_pool.h for existing code

}  // namespace mini_inference
