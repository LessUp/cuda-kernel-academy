#include <gtest/gtest.h>
#include "tensor.h"
#include "common.h"

using namespace mini_inference;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }
};

// ============================================================================
// Tensor Creation Tests
// ============================================================================

TEST_F(TensorTest, CreateEmpty) {
    Tensor t;
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0);
    EXPECT_EQ(t.ndim(), 0);
}

TEST_F(TensorTest, CreateWithShape) {
    Tensor t({2, 3, 4});
    EXPECT_FALSE(t.empty());
    EXPECT_EQ(t.size(), 24);
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.dim(0), 2);
    EXPECT_EQ(t.dim(1), 3);
    EXPECT_EQ(t.dim(2), 4);
}

TEST_F(TensorTest, CreateWithData) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    Tensor t({2, 3}, data.data());
    
    auto result = t.to_host();
    EXPECT_EQ(result.size(), 6);
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(result[i], data[i]);
    }
}

TEST_F(TensorTest, Clone) {
    std::vector<float> data = {1, 2, 3, 4};
    Tensor t({2, 2}, data.data());
    
    Tensor t2 = t.clone();
    
    auto result = t2.to_host();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(result[i], data[i]);
    }
}

TEST_F(TensorTest, Reshape) {
    Tensor t({2, 3, 4});
    t.reshape({6, 4});
    
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.dim(0), 6);
    EXPECT_EQ(t.dim(1), 4);
    EXPECT_EQ(t.size(), 24);
}

TEST_F(TensorTest, ReshapeInvalidSize) {
    Tensor t({2, 3, 4});
    EXPECT_THROW(t.reshape({5, 5}), std::invalid_argument);
}

// ============================================================================
// Tensor Operations Tests
// ============================================================================

TEST_F(TensorTest, Fill) {
    Tensor t({10, 10});
    t.fill(3.14f);
    
    auto result = t.to_host();
    for (float val : result) {
        EXPECT_FLOAT_EQ(val, 3.14f);
    }
}

TEST_F(TensorTest, Zero) {
    std::vector<float> data(100, 1.0f);
    Tensor t({10, 10}, data.data());
    t.zero();
    
    auto result = t.to_host();
    for (float val : result) {
        EXPECT_FLOAT_EQ(val, 0.0f);
    }
}

TEST_F(TensorTest, Add) {
    std::vector<float> a_data = {1, 2, 3, 4};
    std::vector<float> b_data = {5, 6, 7, 8};
    
    Tensor a({2, 2}, a_data.data());
    Tensor b({2, 2}, b_data.data());
    
    Tensor c = add(a, b);
    
    auto result = c.to_host();
    EXPECT_FLOAT_EQ(result[0], 6);
    EXPECT_FLOAT_EQ(result[1], 8);
    EXPECT_FLOAT_EQ(result[2], 10);
    EXPECT_FLOAT_EQ(result[3], 12);
}

TEST_F(TensorTest, AddBias) {
    std::vector<float> x_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> bias_data = {10, 20, 30};
    
    Tensor x({2, 3}, x_data.data());
    Tensor bias({3}, bias_data.data());
    
    Tensor y = add_bias(x, bias);
    
    auto result = y.to_host();
    EXPECT_FLOAT_EQ(result[0], 11);  // 1 + 10
    EXPECT_FLOAT_EQ(result[1], 22);  // 2 + 20
    EXPECT_FLOAT_EQ(result[2], 33);  // 3 + 30
    EXPECT_FLOAT_EQ(result[3], 14);  // 4 + 10
    EXPECT_FLOAT_EQ(result[4], 25);  // 5 + 20
    EXPECT_FLOAT_EQ(result[5], 36);  // 6 + 30
}

TEST_F(TensorTest, ReLU) {
    std::vector<float> data = {-2, -1, 0, 1, 2};
    Tensor x({5}, data.data());
    
    Tensor y = relu(x);
    
    auto result = y.to_host();
    EXPECT_FLOAT_EQ(result[0], 0);
    EXPECT_FLOAT_EQ(result[1], 0);
    EXPECT_FLOAT_EQ(result[2], 0);
    EXPECT_FLOAT_EQ(result[3], 1);
    EXPECT_FLOAT_EQ(result[4], 2);
}

TEST_F(TensorTest, Softmax) {
    std::vector<float> data = {1, 2, 3};
    Tensor x({1, 3}, data.data());
    
    Tensor y = softmax(x);
    
    auto result = y.to_host();
    
    // Check sum is 1
    float sum = result[0] + result[1] + result[2];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
    
    // Check ordering
    EXPECT_LT(result[0], result[1]);
    EXPECT_LT(result[1], result[2]);
}

TEST_F(TensorTest, MatMul) {
    // A: 2x3, B: 3x4 -> C: 2x4
    std::vector<float> a_data(6), b_data(12);
    random_init(a_data.data(), 6);
    random_init(b_data.data(), 12);
    
    Tensor a({2, 3}, a_data.data());
    Tensor b({3, 4}, b_data.data());
    
    Tensor c = matmul(a, b);
    
    EXPECT_EQ(c.ndim(), 2);
    EXPECT_EQ(c.dim(0), 2);
    EXPECT_EQ(c.dim(1), 4);
    
    // Verify against CPU reference
    std::vector<float> c_ref(8);
    cpu_matmul(a_data.data(), b_data.data(), c_ref.data(), 2, 4, 3);
    
    auto c_result = c.to_host();
    float max_error = compare_matrices(c_result.data(), c_ref.data(), 8);
    EXPECT_LT(max_error, 1e-4f);
}

TEST_F(TensorTest, FusedLinearReLU) {
    // X: 4x8, W: 8x16, bias: 16 -> Y: 4x16
    std::vector<float> x_data(32), w_data(128), bias_data(16);
    random_init(x_data.data(), 32, -0.5f, 0.5f);
    random_init(w_data.data(), 128, -0.5f, 0.5f);
    random_init(bias_data.data(), 16, -0.1f, 0.1f);
    
    Tensor x({4, 8}, x_data.data());
    Tensor w({8, 16}, w_data.data());
    Tensor bias({16}, bias_data.data());
    
    Tensor y = fused_linear_relu(x, w, bias);
    
    EXPECT_EQ(y.dim(0), 4);
    EXPECT_EQ(y.dim(1), 16);
    
    // Verify against CPU reference
    std::vector<float> y_ref(64);
    cpu_matmul_bias_relu(x_data.data(), w_data.data(), y_ref.data(),
                         bias_data.data(), 4, 16, 8, true, true);
    
    auto y_result = y.to_host();
    float max_error = compare_matrices(y_result.data(), y_ref.data(), 64);
    EXPECT_LT(max_error, 1e-3f);
}

// ============================================================================
// Memory Pool Tests
// ============================================================================

TEST_F(TensorTest, MemoryPoolBasic) {
    MemoryPool::instance().clear_all();
    
    // Allocate and deallocate
    {
        PooledMemory mem(1024 * sizeof(float));
        EXPECT_NE(mem.get(), nullptr);
    }
    
    // Should be cached now
    auto stats = MemoryPool::instance().get_stats();
    EXPECT_GT(stats.cached_size, 0);
    
    // Allocate again - should hit cache
    {
        PooledMemory mem(1024 * sizeof(float));
        EXPECT_NE(mem.get(), nullptr);
    }
    
    stats = MemoryPool::instance().get_stats();
    EXPECT_GT(stats.cache_hits, 0);
}

TEST_F(TensorTest, MemoryPoolMultiple) {
    MemoryPool::instance().clear_all();
    
    std::vector<PooledMemory> mems;
    for (int i = 0; i < 10; i++) {
        mems.emplace_back(1024 * sizeof(float));
    }
    
    auto stats = MemoryPool::instance().get_stats();
    EXPECT_GE(stats.total_allocated, 10 * 1024 * sizeof(float));
}
