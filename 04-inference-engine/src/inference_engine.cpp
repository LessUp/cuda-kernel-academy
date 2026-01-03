#include "inference_engine.h"

namespace mini_inference {

void InferenceEngine::init(int device_id) {
    if (initialized_) {
        cleanup();
    }
    
    device_id_ = device_id;
    CUDA_CHECK(cudaSetDevice(device_id_));
    
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));
    
    initialized_ = true;
}

void InferenceEngine::cleanup() {
    layers_.clear();
    temp_buffer_.free();
    
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
        cublas_handle_ = nullptr;
    }
    
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    
    initialized_ = false;
}

bool InferenceEngine::load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Read header
    WeightFileHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    if (header.magic != WEIGHT_FILE_MAGIC) {
        return false;
    }
    
    if (header.version != WEIGHT_FILE_VERSION) {
        return false;
    }
    
    layers_.clear();
    layers_.reserve(header.num_layers);
    
    // Read each layer
    for (uint32_t i = 0; i < header.num_layers; i++) {
        LayerMeta meta;
        file.read(reinterpret_cast<char*>(&meta), sizeof(meta));
        
        LayerWeights layer;
        layer.in_features = meta.in_features;
        layer.out_features = meta.out_features;
        layer.has_bias = meta.has_bias != 0;
        
        // Read weights
        size_t weight_size = static_cast<size_t>(meta.in_features) * meta.out_features;
        std::vector<float> weight_data(weight_size);
        file.read(reinterpret_cast<char*>(weight_data.data()), weight_size * sizeof(float));
        
        layer.weights.allocate(weight_size * sizeof(float));
        layer.weights.copy_from_host(weight_data.data(), weight_size * sizeof(float));
        
        // Read bias if present
        if (layer.has_bias) {
            std::vector<float> bias_data(meta.out_features);
            file.read(reinterpret_cast<char*>(bias_data.data()), meta.out_features * sizeof(float));
            
            layer.bias.allocate(meta.out_features * sizeof(float));
            layer.bias.copy_from_host(bias_data.data(), meta.out_features * sizeof(float));
        }
        
        layers_.push_back(std::move(layer));
    }
    
    return true;
}

bool InferenceEngine::save_weights(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Write header
    WeightFileHeader header;
    header.magic = WEIGHT_FILE_MAGIC;
    header.version = WEIGHT_FILE_VERSION;
    header.num_layers = static_cast<uint32_t>(layers_.size());
    std::fill(std::begin(header.reserved), std::end(header.reserved), 0);
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    
    // Write each layer
    for (const auto& layer : layers_) {
        LayerMeta meta;
        meta.type = 0;  // Linear
        meta.in_features = layer.in_features;
        meta.out_features = layer.out_features;
        meta.has_bias = layer.has_bias ? 1 : 0;
        file.write(reinterpret_cast<const char*>(&meta), sizeof(meta));
        
        // Write weights
        size_t weight_size = static_cast<size_t>(layer.in_features) * layer.out_features;
        std::vector<float> weight_data(weight_size);
        layer.weights.copy_to_host(weight_data.data(), weight_size * sizeof(float));
        file.write(reinterpret_cast<const char*>(weight_data.data()), weight_size * sizeof(float));
        
        // Write bias if present
        if (layer.has_bias) {
            std::vector<float> bias_data(layer.out_features);
            layer.bias.copy_to_host(bias_data.data(), layer.out_features * sizeof(float));
            file.write(reinterpret_cast<const char*>(bias_data.data()), layer.out_features * sizeof(float));
        }
    }
    
    return true;
}

void InferenceEngine::add_layer(int in_features, int out_features, bool has_bias,
                                 const float* weights_data, const float* bias_data) {
    LayerWeights layer;
    layer.in_features = in_features;
    layer.out_features = out_features;
    layer.has_bias = has_bias;
    
    size_t weight_size = static_cast<size_t>(in_features) * out_features;
    layer.weights.allocate(weight_size * sizeof(float));
    layer.weights.copy_from_host(weights_data, weight_size * sizeof(float));
    
    if (has_bias && bias_data) {
        layer.bias.allocate(out_features * sizeof(float));
        layer.bias.copy_from_host(bias_data, out_features * sizeof(float));
    }
    
    layers_.push_back(std::move(layer));
}

void InferenceEngine::forward(const float* input, float* output, int batch_size) {
    if (layers_.empty()) {
        throw std::runtime_error("No layers loaded");
    }
    
    // Allocate temp buffer for intermediate results
    size_t max_size = 0;
    for (const auto& layer : layers_) {
        max_size = std::max(max_size, 
            static_cast<size_t>(batch_size) * std::max(layer.in_features, layer.out_features));
    }
    
    if (temp_buffer_.size() < max_size * sizeof(float)) {
        temp_buffer_.allocate(max_size * sizeof(float));
    }
    
    const float* current_input = input;
    float* current_output = (layers_.size() == 1) ? output : temp_buffer_.get();
    
    for (size_t i = 0; i < layers_.size(); i++) {
        const auto& layer = layers_[i];
        bool is_last = (i == layers_.size() - 1);
        bool apply_relu = !is_last;  // ReLU on all but last layer
        
        if (is_last) {
            current_output = output;
        }
        
        // Use fused kernel for MatMul + Bias + ReLU
        launch_fused_gemm(
            current_input,
            layer.weights.get(),
            current_output,
            layer.has_bias ? layer.bias.get() : nullptr,
            batch_size,
            layer.out_features,
            layer.in_features,
            layer.has_bias,
            apply_relu,
            stream_
        );
        
        // Swap buffers for next layer
        if (!is_last) {
            current_input = current_output;
            current_output = (i + 2 == layers_.size()) ? output : temp_buffer_.get();
        }
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void InferenceEngine::forward_with_timing(const float* input, float* output, int batch_size,
                                           std::vector<float>& layer_times_ms) {
    if (layers_.empty()) {
        throw std::runtime_error("No layers loaded");
    }
    
    layer_times_ms.clear();
    layer_times_ms.reserve(layers_.size());
    
    // Allocate temp buffer
    size_t max_size = 0;
    for (const auto& layer : layers_) {
        max_size = std::max(max_size, 
            static_cast<size_t>(batch_size) * std::max(layer.in_features, layer.out_features));
    }
    
    if (temp_buffer_.size() < max_size * sizeof(float)) {
        temp_buffer_.allocate(max_size * sizeof(float));
    }
    
    const float* current_input = input;
    float* current_output = (layers_.size() == 1) ? output : temp_buffer_.get();
    
    GpuTimer timer;
    
    for (size_t i = 0; i < layers_.size(); i++) {
        const auto& layer = layers_[i];
        bool is_last = (i == layers_.size() - 1);
        bool apply_relu = !is_last;
        
        if (is_last) {
            current_output = output;
        }
        
        timer.start(stream_);
        
        launch_fused_gemm(
            current_input,
            layer.weights.get(),
            current_output,
            layer.has_bias ? layer.bias.get() : nullptr,
            batch_size,
            layer.out_features,
            layer.in_features,
            layer.has_bias,
            apply_relu,
            stream_
        );
        
        timer.stop(stream_);
        layer_times_ms.push_back(timer.elapsed_ms());
        
        if (!is_last) {
            current_input = current_output;
            current_output = (i + 2 == layers_.size()) ? output : temp_buffer_.get();
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

void create_random_weights(const std::string& path, 
                           const std::vector<std::pair<int, int>>& layer_dims,
                           bool with_bias) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create weight file: " + path);
    }
    
    WeightFileHeader header;
    header.magic = WEIGHT_FILE_MAGIC;
    header.version = WEIGHT_FILE_VERSION;
    header.num_layers = static_cast<uint32_t>(layer_dims.size());
    std::fill(std::begin(header.reserved), std::end(header.reserved), 0);
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    
    std::mt19937 gen(42);
    
    for (const auto& [in_features, out_features] : layer_dims) {
        LayerMeta meta;
        meta.type = 0;
        meta.in_features = in_features;
        meta.out_features = out_features;
        meta.has_bias = with_bias ? 1 : 0;
        file.write(reinterpret_cast<const char*>(&meta), sizeof(meta));
        
        // Xavier initialization
        float scale = std::sqrt(2.0f / (in_features + out_features));
        std::normal_distribution<float> dist(0.0f, scale);
        
        size_t weight_size = static_cast<size_t>(in_features) * out_features;
        std::vector<float> weights(weight_size);
        for (auto& w : weights) {
            w = dist(gen);
        }
        file.write(reinterpret_cast<const char*>(weights.data()), weight_size * sizeof(float));
        
        if (with_bias) {
            std::vector<float> bias(out_features, 0.0f);
            file.write(reinterpret_cast<const char*>(bias.data()), out_features * sizeof(float));
        }
    }
}

void load_weights_from_host(LayerWeights& layer, 
                            int in_features, int out_features,
                            const float* weights, const float* bias) {
    layer.in_features = in_features;
    layer.out_features = out_features;
    layer.has_bias = (bias != nullptr);
    
    size_t weight_size = static_cast<size_t>(in_features) * out_features;
    layer.weights.allocate(weight_size * sizeof(float));
    layer.weights.copy_from_host(weights, weight_size * sizeof(float));
    
    if (bias) {
        layer.bias.allocate(out_features * sizeof(float));
        layer.bias.copy_from_host(bias, out_features * sizeof(float));
    }
}

} // namespace mini_inference
