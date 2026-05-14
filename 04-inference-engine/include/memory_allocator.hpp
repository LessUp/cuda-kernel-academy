#pragma once

/**
 * @file memory_allocator.hpp
 * @brief Memory allocator interface for decoupling tensor from memory pool
 *
 * Provides an abstract interface for memory allocation, allowing different
 * allocation strategies (pool-based, direct cudaMalloc, unified memory, etc.)
 * to be swapped without modifying Tensor code.
 */

#include <cstddef>

namespace mini_inference {

/**
 * @brief Abstract interface for GPU memory allocation
 *
 * This interface provides a seam for memory allocation, enabling:
 * - Different allocation strategies (pool, direct, unified memory)
 * - Mock allocators for unit testing
 * - Custom memory management for specific use cases
 */
class MemoryAllocator {
public:
    virtual ~MemoryAllocator() = default;

    /**
     * @brief Allocate GPU memory
     * @param bytes Number of bytes to allocate
     * @return Pointer to allocated memory, or nullptr on failure
     */
    virtual void* allocate(size_t bytes) = 0;

    /**
     * @brief Deallocate GPU memory
     * @param ptr Pointer to memory allocated by this allocator
     */
    virtual void deallocate(void* ptr) = 0;

    /**
     * @brief Get the name of this allocator for debugging
     */
    virtual const char* name() const = 0;
};

/**
 * @brief Direct CUDA allocator using cudaMalloc/cudaFree
 *
 * Simple allocator that directly calls CUDA runtime functions.
 * Suitable for cases where memory pool overhead isn't justified.
 */
class DirectAllocator : public MemoryAllocator {
public:
    void* allocate(size_t bytes) override {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        return (err == cudaSuccess) ? ptr : nullptr;
    }

    void deallocate(void* ptr) override {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    const char* name() const override { return "DirectAllocator"; }

    static DirectAllocator& instance() {
        static DirectAllocator allocator;
        return allocator;
    }
};

}  // namespace mini_inference
