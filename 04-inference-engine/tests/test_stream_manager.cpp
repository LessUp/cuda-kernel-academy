#include <gtest/gtest.h>

#include <atomic>
#include <thread>

#include "common.h"
#include "stream_manager.h"

using namespace mini_inference;

class StreamManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices found.";
        }
        CUDA_CHECK(cudaSetDevice(0));
    }

    void TearDown() override { StreamManager::instance().cleanup(); }
};

TEST_F(StreamManagerTest, InitializeStreams) {
    StreamManager::instance().init(4);
    EXPECT_EQ(StreamManager::instance().num_streams(), 4);
}

TEST_F(StreamManagerTest, GetStreamRoundRobin) {
    StreamManager::instance().init(2);

    cudaStream_t s1 = StreamManager::instance().get_stream();
    cudaStream_t s2 = StreamManager::instance().get_stream();
    cudaStream_t s3 = StreamManager::instance().get_stream();

    // s3 should be same as s1 (round-robin)
    EXPECT_EQ(s1, s3);
    EXPECT_NE(s1, s2);
}

TEST_F(StreamManagerTest, GetStreamByIndex) {
    StreamManager::instance().init(4);

    cudaStream_t s0 = StreamManager::instance().get_stream(0);
    cudaStream_t s1 = StreamManager::instance().get_stream(1);
    cudaStream_t s0_again = StreamManager::instance().get_stream(0);

    EXPECT_EQ(s0, s0_again);
    EXPECT_NE(s0, s1);
}

TEST_F(StreamManagerTest, SyncAll) {
    StreamManager::instance().init(4);

    // Launch some work on different streams
    for (int i = 0; i < 4; i++) {
        cudaStream_t stream = StreamManager::instance().get_stream(i);
        // Just record an event to have something to sync
        cudaEvent_t event;
        CUDA_CHECK(cudaEventCreate(&event));
        CUDA_CHECK(cudaEventRecord(event, stream));
        CUDA_CHECK(cudaEventDestroy(event));
    }

    // Should not throw
    EXPECT_NO_THROW(StreamManager::instance().sync_all());
}

TEST_F(StreamManagerTest, Cleanup) {
    StreamManager::instance().init(4);
    EXPECT_EQ(StreamManager::instance().num_streams(), 4);

    StreamManager::instance().cleanup();
    EXPECT_EQ(StreamManager::instance().num_streams(), 0);
}

// ============================================================================
// AsyncOperation Tests
// ============================================================================

TEST_F(StreamManagerTest, AsyncOperationRecord) {
    StreamManager::instance().init(1);
    cudaStream_t stream = StreamManager::instance().get_stream(0);

    AsyncOperation op;
    op.record(stream);

    // Should complete quickly
    op.wait();
    EXPECT_TRUE(op.is_complete());
}

TEST_F(StreamManagerTest, AsyncOperationWaitOn) {
    StreamManager::instance().init(2);
    cudaStream_t stream1 = StreamManager::instance().get_stream(0);
    cudaStream_t stream2 = StreamManager::instance().get_stream(1);

    AsyncOperation op;
    op.record(stream1);

    // Make stream2 wait for stream1
    op.wait_on(stream2);

    // Sync stream2 should also wait for stream1's work
    CUDA_CHECK(cudaStreamSynchronize(stream2));
    EXPECT_TRUE(op.is_complete());
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

TEST_F(StreamManagerTest, ConcurrentGetStream) {
    StreamManager::instance().init(8);

    std::atomic<int> counter{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < 4; t++) {
        threads.emplace_back([&counter]() {
            for (int i = 0; i < 100; i++) {
                cudaStream_t stream = StreamManager::instance().get_stream();
                EXPECT_NE(stream, nullptr);
                counter++;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(counter.load(), 400);
}
