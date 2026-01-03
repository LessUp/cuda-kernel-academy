#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include "05_attention/topk.cuh"
#include "common/tensor.cuh"
#include "../test_utils.hpp"
#include <algorithm>

// Feature: hpc-ai-optimization-lab, Property 12: TopK Correctness
RC_GTEST_PROP(TopKTest, Correctness, ()) {
    auto n = *rc::gen::inRange<int>(10, 256);
    auto k = *rc::gen::inRange<int>(1, std::min(n, 10));
    auto input = *rc::gen::container<std::vector<float>>(n, rc::gen::arbitrary<float>());
    
    // CPU reference
    std::vector<std::pair<float, int>> indexed(n);
    for (int i = 0; i < n; ++i) {
        indexed[i] = {input[i], i};
    }
    std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // GPU implementation
    hpc::Tensor<float> d_input(n);
    hpc::Tensor<float> d_output(k);
    hpc::Tensor<int> d_indices(k);
    d_input.copy_from_host(input);
    
    hpc::attention::topk<float>(
        d_input.data(), d_output.data(), d_indices.data(), 1, n, k);
    cudaDeviceSynchronize();
    
    auto result = d_output.to_host();
    
    // Check that we got k elements
    RC_ASSERT(result.size() == static_cast<size_t>(k));
    
    // Check that top element is correct
    RC_ASSERT(hpc::test::almost_equal(result[0], indexed[0].first));
}

TEST(TopKTest, BasicTest) {
    std::vector<float> input = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};
    int n = input.size();
    int k = 3;
    
    hpc::Tensor<float> d_input(n);
    hpc::Tensor<float> d_output(k);
    hpc::Tensor<int> d_indices(k);
    d_input.copy_from_host(input);
    
    hpc::attention::topk<float>(
        d_input.data(), d_output.data(), d_indices.data(), 1, n, k);
    cudaDeviceSynchronize();
    
    auto result = d_output.to_host();
    EXPECT_EQ(result.size(), k);
    EXPECT_NEAR(result[0], 9.0f, 1e-5f);  // Largest
}
