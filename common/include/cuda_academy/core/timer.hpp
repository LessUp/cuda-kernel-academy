#pragma once
/**
 * @file timer.hpp
 * @brief CUDA event-based timer for GPU performance measurement
 *
 * Provides high-precision timing using CUDA events, which measure
 * actual GPU execution time rather than CPU wall clock time.
 *
 * @example
 * @code
 * #include <cuda_academy/core/timer.hpp>
 *
 * cuda_academy::CudaTimer timer;
 *
 * timer.start();
 * my_kernel<<<grid, block>>>(args);
 * timer.stop();
 *
 * std::cout << "Kernel time: " << timer.elapsed_ms() << " ms" << std::endl;
 * @endcode
 */

#include <cuda_academy/core/cuda_check.hpp>

#include <cstdio>
#include <cuda_runtime.h>
#include <string>

namespace cuda_academy {

// ============================================================================
// CudaTimer - Event-Based GPU Timer
// ============================================================================

/**
 * @brief High-precision GPU timer using CUDA events
 *
 * Measures GPU execution time by recording CUDA events before and after
 * kernel launches. This captures actual GPU time, not CPU wall clock time.
 *
 * Non-copyable but movable.
 */
class CudaTimer {
public:
    /// Create a timer (allocates CUDA events)
    CudaTimer() {
        CA_CUDA_CHECK(cudaEventCreate(&start_));
        CA_CUDA_CHECK(cudaEventCreate(&stop_));
    }

    /// Destroy timer (frees CUDA events)
    ~CudaTimer() {
        if (start_) {
            cudaEventDestroy(start_);
        }
        if (stop_) {
            cudaEventDestroy(stop_);
        }
    }

    // Non-copyable
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    // Movable
    CudaTimer(CudaTimer&& other) noexcept : start_(other.start_), stop_(other.stop_) {
        other.start_ = nullptr;
        other.stop_ = nullptr;
    }

    CudaTimer& operator=(CudaTimer&& other) noexcept {
        if (this != &other) {
            if (start_) {
                cudaEventDestroy(start_);
            }
            if (stop_) {
                cudaEventDestroy(stop_);
            }
            start_ = other.start_;
            stop_ = other.stop_;
            other.start_ = nullptr;
            other.stop_ = nullptr;
        }
        return *this;
    }

    // ------------------------------------------------------------------------
    // Timing Operations
    // ------------------------------------------------------------------------

    /**
     * @brief Record the start event
     * @param stream CUDA stream to record on (default: default stream)
     */
    void start(cudaStream_t stream = nullptr) {
        CA_CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    /**
     * @brief Record the stop event and synchronize
     * @param stream CUDA stream to record on (default: default stream)
     *
     * Note: This blocks until the stop event completes.
     */
    void stop(cudaStream_t stream = nullptr) {
        CA_CUDA_CHECK(cudaEventRecord(stop_, stream));
        CA_CUDA_CHECK(cudaEventSynchronize(stop_));
    }

    /**
     * @brief Get elapsed time in milliseconds
     * @return Time between start() and stop() in milliseconds
     */
    [[nodiscard]] float elapsed_ms() const {
        float ms = 0.0f;
        CA_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

    /**
     * @brief Get elapsed time in seconds
     * @return Time between start() and stop() in seconds
     */
    [[nodiscard]] float elapsed_s() const { return elapsed_ms() / 1000.0f; }

    /**
     * @brief Get elapsed time in microseconds
     * @return Time between start() and stop() in microseconds
     */
    [[nodiscard]] float elapsed_us() const { return elapsed_ms() * 1000.0f; }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

// ============================================================================
// ScopedTimer - RAII Timing Scope
// ============================================================================

/**
 * @brief RAII scoped timer that automatically stops on destruction
 *
 * Useful for timing code blocks without explicit stop() calls.
 *
 * @example
 * @code
 * {
 *     cuda_academy::ScopedTimer timer("my_kernel");
 *     my_kernel<<<grid, block>>>(args);
 * } // Timer automatically stops and prints
 * @endcode
 */
class ScopedTimer {
public:
    /**
     * @brief Create a scoped timer with a name
     * @param name Name to print with timing result (stored internally)
     * @param stream CUDA stream (default: default stream)
     *
     * Note: The name string is copied into the object, ensuring safe use
     * even with temporary strings.
     */
    explicit ScopedTimer(std::string name, cudaStream_t stream = nullptr)
        : name_(std::move(name)), stream_(stream) {
        timer_.start(stream_);
    }

    /// Stop timer and print result
    ~ScopedTimer() {
        timer_.stop(stream_);
        printf("%s: %.3f ms\n", name_.c_str(), timer_.elapsed_ms());
    }

    // Non-copyable, non-movable
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;

    /// Get elapsed time (must call stop() first)
    [[nodiscard]] float elapsed_ms() const { return timer_.elapsed_ms(); }

private:
    std::string name_;
    cudaStream_t stream_;
    CudaTimer timer_;
};

}  // namespace cuda_academy
