#include "inference_engine.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <algorithm>

using namespace mini_inference;

// Simple softmax for output
void softmax(float* data, int n) {
    float max_val = *std::max_element(data, data + n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    for (int i = 0; i < n; i++) {
        data[i] /= sum;
    }
}

int argmax(const float* data, int n) {
    return std::max_element(data, data + n) - data;
}

void print_header() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    Mini-Inference Engine - MNIST Demo                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

int main(int argc, char** argv) {
    try {
        print_header();
        
        CUDA_CHECK(cudaSetDevice(0));
        
        // Create MNIST network: 784 -> 256 -> 128 -> 10
        std::cout << "Creating MNIST network (784 -> 256 -> 128 -> 10)...\n";
        
        const std::string weight_path = "mnist_weights.bin";
        
        // Create random weights for demo (in real use, load trained weights)
        std::vector<std::pair<int, int>> layer_dims = {
            {784, 256},
            {256, 128},
            {128, 10}
        };
        
        std::cout << "Generating random weights for demo...\n";
        create_random_weights(weight_path, layer_dims, true);
        
        // Initialize engine
        InferenceEngine engine;
        engine.init(0);
        
        // Load weights
        std::cout << "Loading weights from " << weight_path << "...\n";
        if (!engine.load_weights(weight_path)) {
            std::cerr << "Failed to load weights!\n";
            return 1;
        }
        
        std::cout << "Network loaded: " << engine.num_layers() << " layers\n";
        std::cout << "Input dimension: " << engine.input_dim() << "\n";
        std::cout << "Output dimension: " << engine.output_dim() << "\n";
        
        // Create random input (simulating MNIST image)
        const int batch_size = 100;
        std::vector<float> input(batch_size * 784);
        random_init(input.data(), input.size(), 0.0f, 1.0f);
        
        // Allocate device memory
        DeviceMemory d_input(batch_size * 784 * sizeof(float));
        DeviceMemory d_output(batch_size * 10 * sizeof(float));
        
        d_input.copy_from_host(input.data(), batch_size * 784 * sizeof(float));
        
        // Warmup
        std::cout << "\nWarming up...\n";
        for (int i = 0; i < 10; i++) {
            engine.forward(d_input.get(), d_output.get(), batch_size);
        }
        
        // Benchmark inference
        std::cout << "Running inference benchmark...\n";
        
        std::vector<float> layer_times;
        const int num_iterations = 100;
        
        GpuTimer timer;
        timer.start();
        
        for (int i = 0; i < num_iterations; i++) {
            engine.forward(d_input.get(), d_output.get(), batch_size);
        }
        
        timer.stop();
        float total_time = timer.elapsed_ms();
        float avg_time = total_time / num_iterations;
        
        // Get per-layer timing
        engine.forward_with_timing(d_input.get(), d_output.get(), batch_size, layer_times);
        
        // Get output
        std::vector<float> output(batch_size * 10);
        d_output.copy_to_host(output.data(), batch_size * 10 * sizeof(float));
        
        // Print results
        std::cout << "\n┌──────────────────────────────────────────────────────────────────────────────┐\n";
        std::cout << "│                              Inference Results                               │\n";
        std::cout << "├──────────────────────────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Batch size: " << batch_size << "\n";
        std::cout << "│ Iterations: " << num_iterations << "\n";
        std::cout << "│ Total time: " << std::fixed << std::setprecision(2) << total_time << " ms\n";
        std::cout << "│ Average time per batch: " << std::fixed << std::setprecision(3) << avg_time << " ms\n";
        std::cout << "│ Throughput: " << std::fixed << std::setprecision(0) 
                  << (batch_size * 1000.0f / avg_time) << " images/sec\n";
        std::cout << "├──────────────────────────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Per-layer timing:\n";
        
        float layer_total = 0.0f;
        for (size_t i = 0; i < layer_times.size(); i++) {
            std::cout << "│   Layer " << i << ": " << std::fixed << std::setprecision(4) 
                      << layer_times[i] << " ms\n";
            layer_total += layer_times[i];
        }
        std::cout << "│   Total: " << std::fixed << std::setprecision(4) << layer_total << " ms\n";
        
        std::cout << "├──────────────────────────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Sample predictions (first 5 images):\n";
        
        for (int i = 0; i < 5; i++) {
            float* logits = &output[i * 10];
            softmax(logits, 10);
            int pred = argmax(logits, 10);
            
            std::cout << "│   Image " << i << ": predicted class " << pred 
                      << " (confidence: " << std::fixed << std::setprecision(2) 
                      << logits[pred] * 100 << "%)\n";
        }
        
        std::cout << "└──────────────────────────────────────────────────────────────────────────────┘\n";
        
        // Cleanup
        engine.cleanup();
        std::remove(weight_path.c_str());
        
        std::cout << "\nDemo complete!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
