# API Reference

## Core Classes

### DeviceMemory

RAII wrapper for GPU memory allocation.

```cpp
#include "common.h"

// Allocate 1MB
DeviceMemory mem(1024 * 1024);

// Copy from host
std::vector<float> host_data(256);
mem.copy_from_host(host_data.data(), 256 * sizeof(float));

// Copy to host
mem.copy_to_host(host_data.data(), 256 * sizeof(float));

// Zero memory
mem.zero();

// Get raw pointer
float* ptr = mem.get();
```

### Tensor

N-dimensional tensor with GPU storage.

```cpp
#include "tensor.h"

// Create tensor
Tensor t({batch, height, width, channels});

// Create with data
std::vector<float> data(100);
Tensor t({10, 10}, data.data());

// Operations
t.fill(1.0f);
t.zero();
Tensor t2 = t.clone();
t.reshape({100});

// Get data
auto host_data = t.to_host();
```

### MemoryPool

Efficient GPU memory allocation with caching.

```cpp
#include "memory_pool.h"

// Allocate from pool
void* ptr = MemoryPool::instance().allocate(1024);

// Return to pool
MemoryPool::instance().deallocate(ptr);

// Get statistics
auto stats = MemoryPool::instance().get_stats();
printf("Hit rate: %.1f%%\n", 
       100.0 * stats.cache_hits / (stats.cache_hits + stats.cache_misses));

// Clear cache
MemoryPool::instance().clear_cache();
```

### StreamManager

CUDA stream management for concurrent execution.

```cpp
#include "stream_manager.h"

// Initialize with 4 streams
StreamManager::instance().init(4);

// Get stream (round-robin)
cudaStream_t stream = StreamManager::instance().get_stream();

// Get specific stream
cudaStream_t stream0 = StreamManager::instance().get_stream(0);

// Synchronize all
StreamManager::instance().sync_all();
```

### InferenceEngine

Neural network inference engine.

```cpp
#include "inference_engine.h"

InferenceEngine engine;
engine.init(0);  // GPU device 0

// Load weights
engine.load_weights("model.bin");

// Forward pass
engine.forward(input_ptr, output_ptr, batch_size);

// With timing
std::vector<float> layer_times;
engine.forward_with_timing(input_ptr, output_ptr, batch_size, layer_times);

engine.cleanup();
```

## GEMM Functions

### Kernel Launch Functions

```cpp
#include "kernels.cuh"

// Naive MatMul
launch_naive_matmul(A, B, C, M, N, K, stream);

// Tiled GEMM
launch_tiled_gemm(A, B, C, M, N, K, stream);

// Coalesced GEMM
launch_coalesced_gemm(A, B, C, M, N, K, stream);

// Double Buffer GEMM
launch_double_buffer_gemm(A, B, C, M, N, K, stream);

// Optimized GEMM (Register Blocked)
launch_optimized_gemm(A, B, C, M, N, K, stream);

// Fused GEMM + Bias + ReLU
launch_fused_gemm(A, B, C, bias, M, N, K, add_bias, apply_relu, stream);

// cuBLAS
launch_cublas_gemm(handle, A, B, C, M, N, K, stream);
```

### Batched GEMM

```cpp
#include "batch_gemm.h"

// Create batch descriptor
BatchGemmDesc desc(M, N, K, batch_size);
for (int i = 0; i < batch_size; i++) {
    desc.add_matrices(A_ptrs[i], B_ptrs[i], C_ptrs[i]);
}

// Launch batched GEMM
launch_batched_gemm_streams(desc, GemmKernelType::REGISTER_BLOCKED);

// Strided batched GEMM
launch_strided_batched_gemm(A, B, C, M, N, K, batch_size);
```

### Half Precision GEMM

```cpp
#include "half_gemm.cuh"

// FP16 GEMM with FP32 accumulation
launch_half_gemm(A_half, B_half, C, M, N, K, stream);
```

## Quantization

```cpp
#include "quantization.h"

// Compute quantization parameters
auto params = compute_quant_params(data, n);

// Quantize
std::vector<int8_t> quantized(n);
quantize_tensor(data, quantized.data(), n, params);

// Dequantize
std::vector<float> dequantized(n);
dequantize_tensor(quantized.data(), dequantized.data(), n, params);

// Per-channel quantization
auto ch_params = compute_per_channel_params(data, rows, cols);
quantize_per_channel(data, quantized.data(), rows, cols, ch_params);

// Calibration
QuantizationCalibrator calibrator;
calibrator.observe(batch1, n1);
calibrator.observe(batch2, n2);
auto final_params = calibrator.get_params();
```

## Configuration

```cpp
#include "config.h"

// Load from file
Config::instance().load_from_file("config.ini");

// Get values
int device = Config::instance().get_int("CUDA_DEVICE", 0);
bool enabled = Config::instance().get_bool("ENABLE_TENSOR_CORES", true);
std::string preset = Config::instance().get("GEMM_PRESET", "medium");

// Set values
Config::instance().set("LOG_LEVEL", "DEBUG");

// GEMM presets
GemmPreset preset = get_gemm_preset("large");
```

## Logging

```cpp
#include "logger.h"

// Set log level
Logger::instance().set_level(LogLevel::DEBUG);

// Log to file
Logger::instance().set_file("app.log");

// Log messages
LOG_TRACE("Trace message");
LOG_DEBUG("Debug: value = %d", value);
LOG_INFO("Info message");
LOG_WARN("Warning: %s", msg);
LOG_ERROR("Error occurred");
LOG_FATAL("Fatal error");
```

## Performance Measurement

```cpp
#include "kernels.cuh"

// GPU Timer
GpuTimer timer;
timer.start();
// ... kernel execution ...
timer.stop();
float ms = timer.elapsed_ms();

// Benchmark kernel
PerfStats stats = benchmark_kernel(
    GemmKernelType::REGISTER_BLOCKED,
    A, B, C, M, N, K,
    5,   // warmup iterations
    20,  // benchmark iterations
    cublas_handle
);
printf("GFLOPS: %.2f\n", stats.gflops);
```

## Profiler

```cpp
#include "profiler.h"

Profiler profiler;

// Profile a kernel
auto result = profiler.profile("MyKernel", M, N, K, warmup, iters, [&]() {
    launch_optimized_gemm(A, B, C, M, N, K);
});

printf("Time: %.3f ms, GFLOPS: %.2f\n", result.time_ms, result.gflops);

// Roofline analysis
RooflineAnalyzer roofline(peak_gflops, peak_bandwidth);
roofline.add_point(result);
roofline.analyze();
```

## Auto-Tuner

```cpp
#include "autotuner.h"

AutoTuner tuner;

// Find best kernel for given dimensions
GemmKernelType best = tuner.find_best_kernel(M, N, K, cublas_handle);

// Get recommended config
GemmConfig config = tuner.get_recommended_config(M, N, K);
```
