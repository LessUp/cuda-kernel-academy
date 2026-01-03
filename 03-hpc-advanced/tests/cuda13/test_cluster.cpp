#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include "07_cuda13_features/cluster.cuh"
#include "common/tensor.cuh"
#include "../test_utils.hpp"
#include <numeric>

// Feature: hpc-ai-optimization-lab, Property 14: Cluster Reduce Correctness
RC_GTEST_PROP(ClusterTest, ReduceCorrectness, ()) {
    auto n = *rc::gen::inRange<size_t>(256, 4096);
    auto input = *rc::gen::container<std::vector<float>>(n,
        rc::gen::map(rc::gen::arbitrary<float>(), [](float x) {
            return std::clamp(x, -1.0f, 1.0f);
        }));
    
    // CPU reference
    float expected = std::accumulate(input.begin(), input.end(), 0.0f);
    
    // GPU implementation
    hpc::Tensor<float> d_input(n);
    hpc::Tensor<float> d_output(1);
    d_input.copy_from_host(input);
    
    hpc::cuda13::ClusterConfig config{{1, 1, 1}, {1, 1, 1}, {256, 1, 1}};
    hpc::cuda13::cluster_reduce<float>(d_input.data(), d_output.data(), n, config);
    cudaDeviceSynchronize();
    
    auto result = d_output.to_host();
    RC_ASSERT(hpc::test::almost_equal(result[0], expected, 1e-2f, 1e-3f));
}

TEST(ClusterTest, BasicReduce) {
    size_t n = 1024;
    std::vector<float> input(n, 1.0f);
    
    hpc::Tensor<float> d_input(n);
    hpc::Tensor<float> d_output(1);
    d_input.copy_from_host(input);
    
    hpc::cuda13::ClusterConfig config{{1, 1, 1}, {1, 1, 1}, {256, 1, 1}};
    hpc::cuda13::cluster_reduce<float>(d_input.data(), d_output.data(), n, config);
    cudaDeviceSynchronize();
    
    auto result = d_output.to_host();
    EXPECT_NEAR(result[0], static_cast<float>(n), 1e-3f);
}
