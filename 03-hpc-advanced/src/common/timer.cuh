#pragma once

#include <cuda_runtime.h>
#include "cuda_check.cuh"

namespace hpc {

class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    // Non-copyable
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    // Movable
    CudaTimer(CudaTimer&& other) noexcept
        : start_(other.start_), stop_(other.stop_) {
        other.start_ = nullptr;
        other.stop_ = nullptr;
    }

    CudaTimer& operator=(CudaTimer&& other) noexcept {
        if (this != &other) {
            if (start_) cudaEventDestroy(start_);
            if (stop_) cudaEventDestroy(stop_);
            start_ = other.start_;
            stop_ = other.stop_;
            other.start_ = nullptr;
            other.stop_ = nullptr;
        }
        return *this;
    }

    void start(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    void stop(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }

    [[nodiscard]] float elapsed_ms() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

} // namespace hpc
