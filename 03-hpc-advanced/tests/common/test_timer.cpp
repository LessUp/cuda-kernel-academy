#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include "common/timer.cuh"
#include "common/tensor.cuh"

// Feature: hpc-ai-optimization-lab, Property 2: Timer Non-Negativity
RC_GTEST_PROP(TimerTest, NonNegativity, ()) {
    hpc::CudaTimer timer;
    
    // Create some work
    auto size = *rc::gen::inRange<size_t>(1024, 1024 * 1024);
    hpc::Tensor<float> tensor(size);
    
    timer.start();
    tensor.zero();  // Some GPU work
    cudaDeviceSynchronize();
    timer.stop();
    
    float elapsed = timer.elapsed_ms();
    RC_ASSERT(elapsed >= 0.0f);
}

TEST(TimerTest, BasicTiming) {
    hpc::CudaTimer timer;
    
    timer.start();
    // Small delay
    hpc::Tensor<float> tensor(1024 * 1024);
    tensor.zero();
    cudaDeviceSynchronize();
    timer.stop();
    
    float elapsed = timer.elapsed_ms();
    EXPECT_GE(elapsed, 0.0f);
}
