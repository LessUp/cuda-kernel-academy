#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include "01_elementwise/vector_add.cuh"
#include "common/tensor.cuh"
#include "../test_utils.hpp"

RC_GTEST_PROP(VectorAddTest, Correctness, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 1024 * 64);
    auto a = *rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>());
    auto b = *rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>());
    
    // CPU reference
    std::vector<float> expected(size);
    for (size_t i = 0; i < size; ++i) {
        expected[i] = a[i] + b[i];
    }
    
    // GPU implementation
    hpc::Tensor<float> d_a(size);
    hpc::Tensor<float> d_b(size);
    hpc::Tensor<float> d_c(size);
    d_a.copy_from_host(a);
    d_b.copy_from_host(b);
    
    hpc::elementwise::vector_add<float, hpc::elementwise::OptLevel::GridStride>(
        d_a.data(), d_b.data(), d_c.data(), size);
    cudaDeviceSynchronize();
    
    auto result = d_c.to_host();
    
    for (size_t i = 0; i < size; ++i) {
        RC_ASSERT(hpc::test::almost_equal(result[i], expected[i]));
    }
}

TEST(VectorAddTest, BasicTest) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};
    std::vector<float> expected = {5.0f, 7.0f, 9.0f};
    
    hpc::Tensor<float> d_a(a.size());
    hpc::Tensor<float> d_b(b.size());
    hpc::Tensor<float> d_c(a.size());
    d_a.copy_from_host(a);
    d_b.copy_from_host(b);
    
    hpc::elementwise::vector_add<float, hpc::elementwise::OptLevel::Naive>(
        d_a.data(), d_b.data(), d_c.data(), a.size());
    cudaDeviceSynchronize();
    
    auto result = d_c.to_host();
    EXPECT_TRUE(hpc::test::vectors_almost_equal(result, expected));
}
