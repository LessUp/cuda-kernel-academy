#pragma once

#include <fstream>

#include "common.h"
#include "kernels.cuh"

namespace mini_inference {

// ============================================================================
// Weight File Format
// ============================================================================

constexpr uint32_t WEIGHT_FILE_MAGIC = 0x4D494E49;  // "MINI"
constexpr uint32_t WEIGHT_FILE_VERSION = 1;

struct WeightFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_layers;
    uint32_t reserved[5];
};

struct LayerMeta {
    uint32_t type;  // 0 = Linear
    uint32_t in_features;
    uint32_t out_features;
    uint32_t has_bias;
};

struct LayerWeights {
    DeviceMemory weights;
    DeviceMemory bias;
    int in_features;
    int out_features;
    bool has_bias;
};

// ============================================================================
// Inference Engine
// ============================================================================

class InferenceEngine {
public:
    InferenceEngine() = default;
    ~InferenceEngine() { cleanup(); }

    // Disable copy
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    // Initialize engine
    void init(int device_id = 0);

    // Load model weights from file
    bool load_weights(const std::string& path);

    // Save model weights to file
    bool save_weights(const std::string& path) const;

    // Add a layer manually
    void add_layer(int in_features, int out_features, bool has_bias, const float* weights_data,
                   const float* bias_data = nullptr);

    // Forward pass
    void forward(const float* input, float* output, int batch_size);

    // Forward pass with timing info
    void forward_with_timing(const float* input, float* output, int batch_size,
                             std::vector<float>& layer_times_ms);

    // Get layer count
    size_t num_layers() const { return layers_.size(); }

    // Get input/output dimensions
    int input_dim() const { return layers_.empty() ? 0 : layers_[0].in_features; }
    int output_dim() const { return layers_.empty() ? 0 : layers_.back().out_features; }

    // Cleanup resources
    void cleanup();

    // Check if initialized
    bool is_initialized() const { return initialized_; }

private:
    std::vector<LayerWeights> layers_;
    cublasHandle_t cublas_handle_ = nullptr;
    cudaStream_t stream_ = nullptr;
    // CKA-002: ping-pong buffers to prevent in-place aliasing across layers
    DeviceMemory temp_a_;
    DeviceMemory temp_b_;
    bool initialized_ = false;
    int device_id_ = 0;
};

// ============================================================================
// Utility Functions for Weight Management
// ============================================================================

// Create random weights for testing
void create_random_weights(const std::string& path,
                           const std::vector<std::pair<int, int>>& layer_dims,
                           bool with_bias = true);

// Load weights from host arrays
void load_weights_from_host(LayerWeights& layer, int in_features, int out_features,
                            const float* weights, const float* bias = nullptr);

}  // namespace mini_inference
