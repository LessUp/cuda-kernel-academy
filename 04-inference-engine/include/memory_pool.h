#pragma once

#include <algorithm>
#include <map>
#include <mutex>
#include <vector>

#include "common.h"

namespace mini_inference {

// ============================================================================
// GPU Memory Pool
// Efficient memory allocation with caching to reduce cudaMalloc overhead
// ============================================================================

class MemoryPool {
public:
    static MemoryPool& instance() {
        static MemoryPool pool;
        return pool;
    }

    // Allocate memory from pool
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Round up to alignment
        size = align_size(size);

        // Check free list for suitable block
        auto it = free_blocks_.lower_bound(size);
        if (it != free_blocks_.end()) {
            void* ptr = it->second;
            free_blocks_.erase(it);
            allocated_blocks_[ptr] = size;
            stats_.cache_hits++;
            return ptr;
        }

        // Allocate new block
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            // Try to free some cached memory
            if (try_free_cached(size)) {
                err = cudaMalloc(&ptr, size);
            }
            if (err != cudaSuccess) {
                throw CudaException(err, __FILE__, __LINE__);
            }
        }

        allocated_blocks_[ptr] = size;
        stats_.total_allocated += size;
        stats_.cache_misses++;
        return ptr;
    }

    // Return memory to pool
    void deallocate(void* ptr) {
        if (!ptr)
            return;

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocated_blocks_.find(ptr);
        if (it == allocated_blocks_.end()) {
            // Not from this pool, free directly
            cudaFree(ptr);
            return;
        }

        size_t size = it->second;
        allocated_blocks_.erase(it);

        // Add to free list
        free_blocks_.insert({size, ptr});
        stats_.cached_size += size;
    }

    // Free all cached memory
    void clear_cache() {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto& [size, ptr] : free_blocks_) {
            cudaFree(ptr);
        }
        free_blocks_.clear();
        stats_.cached_size = 0;
    }

    // Free all memory (including allocated)
    void clear_all() {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto& [ptr, size] : allocated_blocks_) {
            cudaFree(ptr);
        }
        allocated_blocks_.clear();

        for (auto& [size, ptr] : free_blocks_) {
            cudaFree(ptr);
        }
        free_blocks_.clear();

        stats_ = {};
    }

    // Get statistics
    struct Stats {
        size_t total_allocated = 0;
        size_t cached_size = 0;
        size_t cache_hits = 0;
        size_t cache_misses = 0;
    };

    Stats get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

    void print_stats() const {
        auto stats = get_stats();
        printf("Memory Pool Statistics:\n");
        printf("  Total allocated: %.2f MB\n", stats.total_allocated / (1024.0 * 1024.0));
        printf("  Cached size: %.2f MB\n", stats.cached_size / (1024.0 * 1024.0));
        printf("  Cache hits: %zu\n", stats.cache_hits);
        printf("  Cache misses: %zu\n", stats.cache_misses);
        printf("  Hit rate: %.1f%%\n",
               stats.cache_hits + stats.cache_misses > 0
                   ? 100.0 * stats.cache_hits / (stats.cache_hits + stats.cache_misses)
                   : 0.0);
    }

private:
    MemoryPool() = default;
    ~MemoryPool() { clear_all(); }

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    static constexpr size_t ALIGNMENT = 256;  // 256-byte alignment

    size_t align_size(size_t size) const { return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1); }

    bool try_free_cached(size_t needed) {
        size_t freed = 0;
        auto it = free_blocks_.begin();
        while (it != free_blocks_.end() && freed < needed) {
            cudaFree(it->second);
            freed += it->first;
            stats_.cached_size -= it->first;
            it = free_blocks_.erase(it);
        }
        return freed >= needed;
    }

    mutable std::mutex mutex_;
    std::map<void*, size_t> allocated_blocks_;
    std::multimap<size_t, void*> free_blocks_;
    Stats stats_;
};

// ============================================================================
// Pooled Device Memory
// RAII wrapper using memory pool
// ============================================================================

class PooledMemory {
public:
    PooledMemory() = default;

    explicit PooledMemory(size_t bytes) {
        if (bytes > 0) {
            ptr_ = static_cast<float*>(MemoryPool::instance().allocate(bytes));
            size_ = bytes;
        }
    }

    ~PooledMemory() {
        if (ptr_) {
            MemoryPool::instance().deallocate(ptr_);
        }
    }

    // Disable copy
    PooledMemory(const PooledMemory&) = delete;
    PooledMemory& operator=(const PooledMemory&) = delete;

    // Enable move
    PooledMemory(PooledMemory&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    PooledMemory& operator=(PooledMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                MemoryPool::instance().deallocate(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void allocate(size_t bytes) {
        if (ptr_) {
            MemoryPool::instance().deallocate(ptr_);
        }
        ptr_ = nullptr;
        size_ = 0;
        if (bytes > 0) {
            ptr_ = static_cast<float*>(MemoryPool::instance().allocate(bytes));
            size_ = bytes;
        }
    }

    void free() {
        if (ptr_) {
            MemoryPool::instance().deallocate(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    float* get() { return ptr_; }
    const float* get() const { return ptr_; }
    size_t size() const { return size_; }
    bool empty() const { return ptr_ == nullptr; }

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
};

}  // namespace mini_inference
