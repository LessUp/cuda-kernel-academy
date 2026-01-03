#pragma once
/**
 * @file memory_pool.hpp
 * @brief Thread-safe GPU memory pool for reducing allocation overhead
 * 
 * Provides a memory pool that caches freed GPU memory blocks for reuse,
 * significantly reducing the overhead of frequent allocations.
 */

#include "../core/cuda_check.hpp"
#include <unordered_map>
#include <vector>
#include <mutex>
#include <memory>

namespace tensorcraft {

/**
 * @brief Thread-safe GPU memory pool
 * 
 * Features:
 * - Caches freed memory blocks by size
 * - Thread-safe allocation/deallocation
 * - Automatic cleanup on destruction
 * - Configurable alignment
 */
class MemoryPool {
public:
    /// Alignment for all allocations (256 bytes for optimal GPU access)
    static constexpr size_t ALIGNMENT = 256;
    
    /**
     * @brief Get singleton instance
     */
    static MemoryPool& instance() {
        static MemoryPool pool;
        return pool;
    }
    
    /**
     * @brief Allocate memory from pool
     * @param size Size in bytes
     * @return Pointer to allocated memory
     */
    void* allocate(size_t size) {
        if (size == 0) return nullptr;
        
        // Round up to alignment
        size = align_size(size);
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Check free list for this size
        auto& free_list = free_blocks_[size];
        if (!free_list.empty()) {
            void* ptr = free_list.back();
            free_list.pop_back();
            stats_.cache_hits++;
            return ptr;
        }
        
        // Allocate new block
        void* ptr = nullptr;
        TC_CUDA_CHECK(cudaMalloc(&ptr, size));
        allocated_sizes_[ptr] = size;
        stats_.allocations++;
        stats_.total_allocated += size;
        
        return ptr;
    }
    
    /**
     * @brief Return memory to pool
     * @param ptr Pointer to memory (can be nullptr)
     */
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = allocated_sizes_.find(ptr);
        if (it != allocated_sizes_.end()) {
            free_blocks_[it->second].push_back(ptr);
            stats_.deallocations++;
        }
    }
    
    /**
     * @brief Free all cached memory
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& [size, blocks] : free_blocks_) {
            for (void* ptr : blocks) {
                cudaFree(ptr);
                stats_.total_freed += size;
            }
        }
        free_blocks_.clear();
        allocated_sizes_.clear();
    }
    
    /**
     * @brief Trim cache to reduce memory usage
     * @param max_cached_bytes Maximum bytes to keep cached
     */
    void trim(size_t max_cached_bytes = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        size_t cached_bytes = 0;
        for (const auto& [size, blocks] : free_blocks_) {
            cached_bytes += size * blocks.size();
        }
        
        // Free blocks until under limit
        while (cached_bytes > max_cached_bytes && !free_blocks_.empty()) {
            // Find largest block to free
            size_t max_size = 0;
            for (const auto& [size, blocks] : free_blocks_) {
                if (!blocks.empty() && size > max_size) {
                    max_size = size;
                }
            }
            
            if (max_size == 0) break;
            
            auto& blocks = free_blocks_[max_size];
            if (!blocks.empty()) {
                void* ptr = blocks.back();
                blocks.pop_back();
                
                auto it = allocated_sizes_.find(ptr);
                if (it != allocated_sizes_.end()) {
                    allocated_sizes_.erase(it);
                }
                
                cudaFree(ptr);
                cached_bytes -= max_size;
                stats_.total_freed += max_size;
            }
        }
    }
    
    /**
     * @brief Get pool statistics
     */
    struct Stats {
        size_t allocations = 0;
        size_t deallocations = 0;
        size_t cache_hits = 0;
        size_t total_allocated = 0;
        size_t total_freed = 0;
    };
    
    Stats get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }
    
    /**
     * @brief Get current cached memory size
     */
    size_t cached_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t total = 0;
        for (const auto& [size, blocks] : free_blocks_) {
            total += size * blocks.size();
        }
        return total;
    }
    
    /// Destructor - frees all memory
    ~MemoryPool() {
        clear();
    }

private:
    MemoryPool() = default;
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    
    static size_t align_size(size_t size) {
        return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }
    
    mutable std::mutex mutex_;
    std::unordered_map<size_t, std::vector<void*>> free_blocks_;
    std::unordered_map<void*, size_t> allocated_sizes_;
    Stats stats_;
};

/**
 * @brief RAII wrapper for pool-allocated memory
 */
template<typename T>
class PoolPtr {
public:
    PoolPtr() = default;
    
    explicit PoolPtr(size_t count) 
        : ptr_(static_cast<T*>(MemoryPool::instance().allocate(count * sizeof(T))))
        , count_(count) {}
    
    ~PoolPtr() {
        if (ptr_) {
            MemoryPool::instance().deallocate(ptr_);
        }
    }
    
    // Move only
    PoolPtr(PoolPtr&& other) noexcept 
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    
    PoolPtr& operator=(PoolPtr&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                MemoryPool::instance().deallocate(ptr_);
            }
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    
    PoolPtr(const PoolPtr&) = delete;
    PoolPtr& operator=(const PoolPtr&) = delete;
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t count() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }
    
    T* operator->() { return ptr_; }
    const T* operator->() const { return ptr_; }
    
    explicit operator bool() const { return ptr_ != nullptr; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

/**
 * @brief Pinned (page-locked) memory allocator for faster CPU-GPU transfers
 */
class PinnedMemory {
public:
    /**
     * @brief Allocate pinned host memory
     */
    static void* allocate(size_t size) {
        void* ptr = nullptr;
        TC_CUDA_CHECK(cudaMallocHost(&ptr, size));
        return ptr;
    }
    
    /**
     * @brief Free pinned host memory
     */
    static void deallocate(void* ptr) {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }
};

/**
 * @brief RAII wrapper for pinned memory
 */
template<typename T>
class PinnedPtr {
public:
    PinnedPtr() = default;
    
    explicit PinnedPtr(size_t count)
        : ptr_(static_cast<T*>(PinnedMemory::allocate(count * sizeof(T))))
        , count_(count) {}
    
    ~PinnedPtr() {
        PinnedMemory::deallocate(ptr_);
    }
    
    // Move only
    PinnedPtr(PinnedPtr&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    
    PinnedPtr& operator=(PinnedPtr&& other) noexcept {
        if (this != &other) {
            PinnedMemory::deallocate(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    
    PinnedPtr(const PinnedPtr&) = delete;
    PinnedPtr& operator=(const PinnedPtr&) = delete;
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }
    size_t count() const { return count_; }
    
    explicit operator bool() const { return ptr_ != nullptr; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

} // namespace tensorcraft
