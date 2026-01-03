#include <gtest/gtest.h>
#include "inference_engine.h"

using namespace mini_inference;

class InferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
        engine_.init(0);
    }
    
    void TearDown() override {
        engine_.cleanup();
    }
    
    InferenceEngine engine_;
};

// ============================================================================
// Property 4: Weight Serialization Round-Trip
// ============================================================================

TEST_F(InferenceTest, WeightSerializationRoundTrip) {
    // Feature: mini-inference-engine, Property 4: Weight Serialization Round-Trip
    const std::string test_path = "/tmp/test_weights.bin";
    
    // Create random weights
    std::vector<std::pair<int, int>> layer_dims = {
        {784, 256},
        {256, 128},
        {128, 10}
    };
    
    create_random_weights(test_path, layer_dims, true);
    
    // Load weights
    ASSERT_TRUE(engine_.load_weights(test_path));
    EXPECT_EQ(engine_.num_layers(), 3);
    EXPECT_EQ(engine_.input_dim(), 784);
    EXPECT_EQ(engine_.output_dim(), 10);
    
    // Save weights to another file
    const std::string test_path2 = "/tmp/test_weights2.bin";
    ASSERT_TRUE(engine_.save_weights(test_path2));
    
    // Load into another engine and compare
    InferenceEngine engine2;
    engine2.init(0);
    ASSERT_TRUE(engine2.load_weights(test_path2));
    
    EXPECT_EQ(engine_.num_layers(), engine2.num_layers());
    EXPECT_EQ(engine_.input_dim(), engine2.input_dim());
    EXPECT_EQ(engine_.output_dim(), engine2.output_dim());
    
    engine2.cleanup();
    
    // Cleanup test files
    std::remove(test_path.c_str());
    std::remove(test_path2.c_str());
}

// ============================================================================
// Property 5: Multi-Layer Forward Pass Consistency
// ============================================================================

TEST_F(InferenceTest, MultiLayerForwardPass) {
    // Feature: mini-inference-engine, Property 5: Multi-Layer Forward Pass Consistency
    const int batch_size = 32;
    const int in_features = 784;
    const int hidden1 = 256;
    const int hidden2 = 128;
    const int out_features = 10;
    
    // Create weights
    std::vector<float> w1(in_features * hidden1);
    std::vector<float> b1(hidden1);
    std::vector<float> w2(hidden1 * hidden2);
    std::vector<float> b2(hidden2);
    std::vector<float> w3(hidden2 * out_features);
    std::vector<float> b3(out_features);
    
    random_init(w1.data(), w1.size(), -0.1f, 0.1f);
    random_init(b1.data(), b1.size(), -0.1f, 0.1f);
    random_init(w2.data(), w2.size(), -0.1f, 0.1f);
    random_init(b2.data(), b2.size(), -0.1f, 0.1f);
    random_init(w3.data(), w3.size(), -0.1f, 0.1f);
    random_init(b3.data(), b3.size(), -0.1f, 0.1f);
    
    // Add layers to engine
    engine_.add_layer(in_features, hidden1, true, w1.data(), b1.data());
    engine_.add_layer(hidden1, hidden2, true, w2.data(), b2.data());
    engine_.add_layer(hidden2, out_features, true, w3.data(), b3.data());
    
    // Create input
    std::vector<float> input(batch_size * in_features);
    random_init(input.data(), input.size(), 0.0f, 1.0f);
    
    // GPU forward pass
    DeviceMemory d_input(batch_size * in_features * sizeof(float));
    DeviceMemory d_output(batch_size * out_features * sizeof(float));
    d_input.copy_from_host(input.data(), batch_size * in_features * sizeof(float));
    
    engine_.forward(d_input.get(), d_output.get(), batch_size);
    
    std::vector<float> gpu_output(batch_size * out_features);
    d_output.copy_to_host(gpu_output.data(), batch_size * out_features * sizeof(float));
    
    // CPU reference forward pass
    std::vector<float> h1(batch_size * hidden1);
    std::vector<float> h2(batch_size * hidden2);
    std::vector<float> cpu_output(batch_size * out_features);
    
    // Layer 1: input -> hidden1 with ReLU
    cpu_matmul_bias_relu(input.data(), w1.data(), h1.data(),
                         b1.data(), batch_size, hidden1, in_features, true, true);
    
    // Layer 2: hidden1 -> hidden2 with ReLU
    cpu_matmul_bias_relu(h1.data(), w2.data(), h2.data(),
                         b2.data(), batch_size, hidden2, hidden1, true, true);
    
    // Layer 3: hidden2 -> output (no ReLU)
    cpu_matmul_bias_relu(h2.data(), w3.data(), cpu_output.data(),
                         b3.data(), batch_size, out_features, hidden2, true, false);
    
    // Compare
    float max_error = compare_matrices(gpu_output.data(), cpu_output.data(), 
                                       batch_size * out_features);
    EXPECT_LT(max_error, 1e-3f) << "Multi-layer forward pass error: " << max_error;
}

// ============================================================================
// Forward Pass with Timing
// ============================================================================

TEST_F(InferenceTest, ForwardWithTiming) {
    const int batch_size = 64;
    
    // Create a simple network
    std::vector<float> w1(784 * 256), b1(256);
    std::vector<float> w2(256 * 10), b2(10);
    
    random_init(w1.data(), w1.size());
    random_init(b1.data(), b1.size());
    random_init(w2.data(), w2.size());
    random_init(b2.data(), b2.size());
    
    engine_.add_layer(784, 256, true, w1.data(), b1.data());
    engine_.add_layer(256, 10, true, w2.data(), b2.data());
    
    std::vector<float> input(batch_size * 784);
    random_init(input.data(), input.size());
    
    DeviceMemory d_input(batch_size * 784 * sizeof(float));
    DeviceMemory d_output(batch_size * 10 * sizeof(float));
    d_input.copy_from_host(input.data(), batch_size * 784 * sizeof(float));
    
    std::vector<float> layer_times;
    engine_.forward_with_timing(d_input.get(), d_output.get(), batch_size, layer_times);
    
    EXPECT_EQ(layer_times.size(), 2);
    for (float t : layer_times) {
        EXPECT_GT(t, 0.0f);
    }
}

// ============================================================================
// Error Handling
// ============================================================================

TEST_F(InferenceTest, LoadNonExistentFile) {
    EXPECT_FALSE(engine_.load_weights("/nonexistent/path/weights.bin"));
}

TEST_F(InferenceTest, ForwardWithNoLayers) {
    DeviceMemory d_input(100 * sizeof(float));
    DeviceMemory d_output(10 * sizeof(float));
    
    EXPECT_THROW(engine_.forward(d_input.get(), d_output.get(), 1), std::runtime_error);
}
