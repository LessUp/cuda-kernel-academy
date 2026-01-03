#include <gtest/gtest.h>
#include "memory_pool.h"
#include "common.h"
#include <thread>
#include <vector>

using namespace mini_inference;

class MemoryPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
        MemoryPool::instance().clear_all();
    }
    
    void TearDown() override {
        MemoryPool::instance().clear_all();
    }
};

// ============================================================================
// Basic Allocation Tests
// ============================================================================

TEST_F(MemoryPoolTest, AllocateAndDeallocate) {
    void* ptr = MemoryPool::instance().allocate(1024);
    EXPECT_NE(ptr, nullptr);
    MemoryPool::instance().deallocate(ptr);
}

TEST_F(MemoryPoolTest, AllocateZeroBytes) {
    // Zero allocation should still work (returns aligned minimum)
    void* ptr = MemoryPool::instance().allocate(0);
    // Implementation may return nullptr or valid pointer for 0 bytes
    MemoryPool::instance().deallocate(ptr);
}

TEST_F(MemoryPoolTest, AllocateLargeBlock) {
    size_t large_size = 256 * 1024 * 1024;  // 256 MB
    void* ptr = MemoryPool::instance().allocate(large_size);
    EXPECT_NE(ptr, nullptr);
    MemoryPool::instance().deallocate(ptr);
}

TEST_F(MemoryPoolTest, MultipleAllocations) {
    std::vector<void*> ptrs;
    for (int i = 0; i < 100; i++) {
        void* ptr = MemoryPool::instance().allocate(1024 * (i + 1));
        EXPECT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }
    
    for (void* ptr : ptrs) {
        MemoryPool::instance().deallocate(ptr);
    }
}

// ============================================================================
// Cache Behavior Tests
// ============================================================================

TEST_F(MemoryPoolTest, CacheHit) {
    // Allocate and deallocate
    void* ptr1 = MemoryPool::instance().allocate(4096);
    MemoryPool::instance().deallocate(ptr1);
    
    auto stats_before = MemoryPool::instance().get_stats();
    
    // Allocate same size - should hit cache
    void* ptr2 = MemoryPool::instance().allocate(4096);
    
    auto stats_after = MemoryPool::instance().get_stats();
    EXPECT_GT(stats_after.cache_hits, stats_before.cache_hits);
    
    MemoryPool::instance().deallocate(ptr2);
}

TEST_F(MemoryPoolTest, CacheMiss) {
    auto stats_before = MemoryPool::instance().get_stats();
    
    // First allocation - cache miss
    void* ptr = MemoryPool::instance().allocate(8192);
    
    auto stats_after = MemoryPool::instance().get_stats();
    EXPECT_GT(stats_after.cache_misses, stats_before.cache_misses);
    
    MemoryPool::instance().deallocate(ptr);
}

TEST_F(MemoryPoolTest, ClearCache) {
    // Allocate and deallocate to populate cache
    void* ptr = MemoryPool::instance().allocate(4096);
    MemoryPool::instance().deallocate(ptr);
    
    auto stats_before = MemoryPool::instance().get_stats();
    EXPECT_GT(stats_before.cached_size, 0);
    
    // Clear cache
    MemoryPool::instance().clear_cache();
    
    auto stats_after = MemoryPool::instance().get_stats();
    EXPECT_EQ(stats_after.cached_size, 0);
}

TEST_F(MemoryPoolTest, ClearAll) {
    // Allocate some memory
    void* ptr1 = MemoryPool::instance().allocate(4096);
    void* ptr2 = MemoryPool::instance().allocate(8192);
    MemoryPool::instance().deallocate(ptr1);  // This goes to cache
    
    // Clear all
    MemoryPool::instance().clear_all();
    
    auto stats = MemoryPool::instance().get_stats();
    EXPECT_EQ(stats.cached_size, 0);
    // Note: ptr2 is now invalid, don't use it
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(MemoryPoolTest, StatisticsTracking) {
    auto initial_stats = MemoryPool::instance().get_stats();
    
    void* ptr1 = MemoryPool::instance().allocate(1024);
    void* ptr2 = MemoryPool::instance().allocate(2048);
    
    auto stats = MemoryPool::instance().get_stats();
    EXPECT_GE(stats.total_allocated, initial_stats.total_allocated + 1024 + 2048);
    
    MemoryPool::instance().deallocate(ptr1);
    MemoryPool::instance().deallocate(ptr2);
}

TEST_F(MemoryPoolTest, HitRateCalculation) {
    // Create some cache hits and misses
    void* ptr1 = MemoryPool::instance().allocate(4096);  // miss
    MemoryPool::instance().deallocate(ptr1);
    
    void* ptr2 = MemoryPool::instance().allocate(4096);  // hit
    MemoryPool::instance().deallocate(ptr2);
    
    void* ptr3 = MemoryPool::instance().allocate(4096);  // hit
    MemoryPool::instance().deallocate(ptr3);
    
    auto stats = MemoryPool::instance().get_stats();
    EXPECT_GE(stats.cache_hits, 2);
    EXPECT_GE(stats.cache_misses, 1);
}

// ============================================================================
// PooledMemory RAII Tests
// ============================================================================

TEST_F(MemoryPoolTest, PooledMemoryBasic) {
    PooledMemory mem(1024 * sizeof(float));
    EXPECT_NE(mem.get(), nullptr);
    EXPECT_EQ(mem.size(), 1024 * sizeof(float));
    EXPECT_FALSE(mem.empty());
}

TEST_F(MemoryPoolTest, PooledMemoryEmpty) {
    PooledMemory mem;
    EXPECT_EQ(mem.get(), nullptr);
    EXPECT_EQ(mem.size(), 0);
    EXPECT_TRUE(mem.empty());
}

TEST_F(MemoryPoolTest, PooledMemoryMove) {
    PooledMemory mem1(1024 * sizeof(float));
    float* ptr = mem1.get();
    
    PooledMemory mem2 = std::move(mem1);
    
    EXPECT_EQ(mem1.get(), nullptr);
    EXPECT_EQ(mem2.get(), ptr);
}

TEST_F(MemoryPoolTest, PooledMemoryMoveAssign) {
    PooledMemory mem1(1024 * sizeof(float));
    PooledMemory mem2(2048 * sizeof(float));
    
    float* ptr1 = mem1.get();
    
    mem2 = std::move(mem1);
    
    EXPECT_EQ(mem1.get(), nullptr);
    EXPECT_EQ(mem2.get(), ptr1);
}

TEST_F(MemoryPoolTest, PooledMemoryReallocate) {
    PooledMemory mem(1024 * sizeof(float));
    float* ptr1 = mem.get();
    
    mem.allocate(2048 * sizeof(float));
    
    EXPECT_NE(mem.get(), nullptr);
    EXPECT_EQ(mem.size(), 2048 * sizeof(float));
}

TEST_F(MemoryPoolTest, PooledMemoryFree) {
    PooledMemory mem(1024 * sizeof(float));
    EXPECT_FALSE(mem.empty());
    
    mem.free();
    
    EXPECT_TRUE(mem.empty());
    EXPECT_EQ(mem.get(), nullptr);
}

TEST_F(MemoryPoolTest, PooledMemoryCopyFromHost) {
    std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
    PooledMemory mem(4 * sizeof(float));
    
    mem.copy_from_host(host_data.data(), 4 * sizeof(float));
    
    std::vector<float> result(4);
    mem.copy_to_host(result.data(), 4 * sizeof(float));
    
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(result[i], host_data[i]);
    }
}

TEST_F(MemoryPoolTest, PooledMemoryZero) {
    std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
    PooledMemory mem(4 * sizeof(float));
    mem.copy_from_host(host_data.data(), 4 * sizeof(float));
    
    mem.zero();
    
    std::vector<float> result(4);
    mem.copy_to_host(result.data(), 4 * sizeof(float));
    
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(result[i], 0.0f);
    }
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(MemoryPoolTest, ConcurrentAllocations) {
    const int num_threads = 4;
    const int allocs_per_thread = 100;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<void*>> thread_ptrs(num_threads);
    
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([t, &thread_ptrs, allocs_per_thread]() {
            for (int i = 0; i < allocs_per_thread; i++) {
                void* ptr = MemoryPool::instance().allocate(1024 * (i % 10 + 1));
                thread_ptrs[t].push_back(ptr);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify all allocations succeeded
    for (int t = 0; t < num_threads; t++) {
        EXPECT_EQ(thread_ptrs[t].size(), allocs_per_thread);
        for (void* ptr : thread_ptrs[t]) {
            EXPECT_NE(ptr, nullptr);
            MemoryPool::instance().deallocate(ptr);
        }
    }
}

TEST_F(MemoryPoolTest, ConcurrentAllocDeallocate) {
    const int num_threads = 4;
    const int iterations = 50;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([iterations]() {
            for (int i = 0; i < iterations; i++) {
                void* ptr = MemoryPool::instance().allocate(4096);
                EXPECT_NE(ptr, nullptr);
                MemoryPool::instance().deallocate(ptr);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}
