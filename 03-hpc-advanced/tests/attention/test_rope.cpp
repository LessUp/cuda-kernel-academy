#include <gtest/gtest.h>
#include "05_attention/rope.cuh"
#include "common/tensor.cuh"
#include "../test_utils.hpp"
#include <cmath>

TEST(RoPETest, BasicTest) {
    int batch = 1, heads = 1, seq = 32, dim = 64;
    int total = batch * heads * seq * dim;
    int half_dim = dim / 2;
    
    auto query = hpc::test::random_vector<float>(total, -1.0f, 1.0f);
    auto key = hpc::test::random_vector<float>(total, -1.0f, 1.0f);
    
    // Generate cos/sin cache
    std::vector<float> cos_cache(seq * half_dim);
    std::vector<float> sin_cache(seq * half_dim);
    for (int s = 0; s < seq; ++s) {
        for (int d = 0; d < half_dim; ++d) {
            float freq = 1.0f / std::pow(10000.0f, 2.0f * d / dim);
            cos_cache[s * half_dim + d] = std::cos(s * freq);
            sin_cache[s * half_dim + d] = std::sin(s * freq);
        }
    }
    
    hpc::Tensor<float> d_query(total);
    hpc::Tensor<float> d_key(total);
    hpc::Tensor<float> d_cos(seq * half_dim);
    hpc::Tensor<float> d_sin(seq * half_dim);
    
    d_query.copy_from_host(query);
    d_key.copy_from_host(key);
    d_cos.copy_from_host(cos_cache);
    d_sin.copy_from_host(sin_cache);
    
    hpc::attention::apply_rope<float>(
        d_query.data(), d_key.data(),
        batch, heads, seq, dim,
        d_cos.data(), d_sin.data());
    cudaDeviceSynchronize();
    
    auto result_q = d_query.to_host();
    EXPECT_EQ(result_q.size(), total);
}
