#include <gtest/gtest.h>
#include "05_attention/flash_attention.cuh"
#include "common/tensor.cuh"
#include "../test_utils.hpp"

TEST(FlashAttentionTest, BasicTest) {
    int batch = 1, heads = 1, seq = 64, dim = 64;
    int total = batch * heads * seq * dim;
    
    auto Q = hpc::test::random_vector<float>(total, -1.0f, 1.0f);
    auto K = hpc::test::random_vector<float>(total, -1.0f, 1.0f);
    auto V = hpc::test::random_vector<float>(total, -1.0f, 1.0f);
    
    hpc::Tensor<float> d_Q(total);
    hpc::Tensor<float> d_K(total);
    hpc::Tensor<float> d_V(total);
    hpc::Tensor<float> d_O(total);
    
    d_Q.copy_from_host(Q);
    d_K.copy_from_host(K);
    d_V.copy_from_host(V);
    
    hpc::attention::FlashAttnConfig config{
        batch, heads, seq, dim,
        1.0f / std::sqrt(static_cast<float>(dim)),
        false
    };
    
    hpc::attention::flash_attention_forward<float>(
        d_Q.data(), d_K.data(), d_V.data(), d_O.data(), config);
    cudaDeviceSynchronize();
    
    auto O = d_O.to_host();
    EXPECT_EQ(O.size(), total);
}
