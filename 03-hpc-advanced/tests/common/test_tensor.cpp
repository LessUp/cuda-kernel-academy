#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include "common/tensor.cuh"
#include "../test_utils.hpp"

// Feature: hpc-ai-optimization-lab, Property 1: Tensor Host-Device Round Trip
RC_GTEST_PROP(TensorTest, HostDeviceRoundTrip, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 1024 * 64);
    auto input = *rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>());
    
    hpc::Tensor<float> tensor(size);
    tensor.copy_from_host(input);
    auto output = tensor.to_host();
    
    RC_ASSERT(input.size() == output.size());
    for (size_t i = 0; i < input.size(); ++i) {
        RC_ASSERT(input[i] == output[i]);
    }
}

TEST(TensorTest, BasicAllocation) {
    hpc::Tensor<float> tensor(1024);
    EXPECT_EQ(tensor.size(), 1024);
    EXPECT_EQ(tensor.bytes(), 1024 * sizeof(float));
    EXPECT_NE(tensor.data(), nullptr);
}

TEST(TensorTest, MoveSemantics) {
    hpc::Tensor<float> tensor1(1024);
    float* ptr = tensor1.data();
    
    hpc::Tensor<float> tensor2 = std::move(tensor1);
    EXPECT_EQ(tensor2.data(), ptr);
    EXPECT_EQ(tensor1.data(), nullptr);
    EXPECT_EQ(tensor1.size(), 0);
}

TEST(TensorTest, ZeroFill) {
    hpc::Tensor<float> tensor(1024);
    tensor.zero();
    
    auto host_data = tensor.to_host();
    for (float val : host_data) {
        EXPECT_EQ(val, 0.0f);
    }
}
