#pragma once

#include <mutex>
#include <queue>
#include <vector>

#include "common.h"

namespace mini_inference {

// ============================================================================
// CUDA Stream Manager
// Manages multiple CUDA streams for concurrent kernel execution
// ============================================================================

class StreamManager {
public:
    static StreamManager& instance() {
        static StreamManager manager;
        return manager;
    }

    // Initialize with specified number of streams
    void init(int num_streams = 4) {
        std::lock_guard<std::mutex> lock(mutex_);
        init_locked(num_streams);
    }

    // Get next stream in round-robin fashion
    cudaStream_t get_stream() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (streams_.empty()) {
            init_locked();
        }
        cudaStream_t stream = streams_[current_stream_];
        current_stream_ = (current_stream_ + 1) % num_streams_;
        return stream;
    }

    // Get specific stream by index
    cudaStream_t get_stream(int index) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (streams_.empty()) {
            init_locked();
        }
        int idx = ((index % num_streams_) + num_streams_) % num_streams_;
        return streams_[idx];
    }

    // Synchronize all streams
    void sync_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& stream : streams_) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    // Synchronize specific stream
    void sync(int index) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < static_cast<int>(streams_.size())) {
            CUDA_CHECK(cudaStreamSynchronize(streams_[index]));
        }
    }

    // Get number of streams
    int num_streams() const { return num_streams_; }

    // Cleanup
    void cleanup() {
        for (auto& stream : streams_) {
            if (stream) {
                cudaStreamDestroy(stream);
            }
        }
        streams_.clear();
        num_streams_ = 0;
        current_stream_ = 0;
    }

private:
    StreamManager() = default;
    ~StreamManager() { cleanup(); }

    StreamManager(const StreamManager&) = delete;
    StreamManager& operator=(const StreamManager&) = delete;

    // Must be called with mutex_ held.
    void init_locked(int num_streams = 4) {
        cleanup();

        num_streams_ = num_streams;
        streams_.resize(num_streams);
        for (int i = 0; i < num_streams; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams_[i]));
        }
        current_stream_ = 0;
    }

    std::vector<cudaStream_t> streams_;
    int num_streams_ = 0;
    int current_stream_ = 0;
    std::mutex mutex_;
};

// ============================================================================
// Async Operation Helper
// ============================================================================

class AsyncOperation {
public:
    AsyncOperation() { CUDA_CHECK(cudaEventCreate(&event_)); }

    ~AsyncOperation() {
        if (event_) {
            cudaEventDestroy(event_);
        }
    }

    // Record completion event
    void record(cudaStream_t stream) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
        stream_ = stream;
        recorded_ = true;
    }

    // Wait for completion
    void wait() {
        if (recorded_) {
            CUDA_CHECK(cudaEventSynchronize(event_));
        }
    }

    // Check if completed (non-blocking)
    bool is_complete() {
        if (!recorded_)
            return true;
        cudaError_t status = cudaEventQuery(event_);
        return status == cudaSuccess;
    }

    // Make another stream wait for this operation
    void wait_on(cudaStream_t other_stream) {
        if (recorded_) {
            CUDA_CHECK(cudaStreamWaitEvent(other_stream, event_, 0));
        }
    }

private:
    cudaEvent_t event_ = nullptr;
    cudaStream_t stream_ = nullptr;
    bool recorded_ = false;
};

}  // namespace mini_inference
