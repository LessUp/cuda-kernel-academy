#pragma once

#include "common.h"
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

namespace mini_inference {

// ============================================================================
// Performance Profiler
// Detailed performance analysis for GEMM kernels
// ============================================================================

struct ProfileResult {
    std::string name;
    float avg_time_ms = 0.0f;
    float min_time_ms = 0.0f;
    float max_time_ms = 0.0f;
    float std_dev_ms = 0.0f;
    float gflops = 0.0f;
    float memory_bandwidth_gb = 0.0f;
    float arithmetic_intensity = 0.0f;  // FLOPs per byte
    float occupancy = 0.0f;
    int iterations = 0;
};

class Profiler {
public:
    Profiler() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~Profiler() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    // Profile a kernel with detailed statistics
    template<typename KernelFunc, typename... Args>
    ProfileResult profile(const std::string& name, 
                          int M, int N, int K,
                          int warmup_iters, int bench_iters,
                          KernelFunc kernel, Args... args) {
        ProfileResult result;
        result.name = name;
        result.iterations = bench_iters;
        
        std::vector<float> times;
        times.reserve(bench_iters);
        
        // Warmup
        for (int i = 0; i < warmup_iters; i++) {
            kernel(args...);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark
        for (int i = 0; i < bench_iters; i++) {
            CUDA_CHECK(cudaEventRecord(start_));
            kernel(args...);
            CUDA_CHECK(cudaEventRecord(stop_));
            CUDA_CHECK(cudaEventSynchronize(stop_));
            
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
            times.push_back(ms);
        }
        
        // Compute statistics
        result.avg_time_ms = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
        result.min_time_ms = *std::min_element(times.begin(), times.end());
        result.max_time_ms = *std::max_element(times.begin(), times.end());
        
        // Standard deviation
        float sq_sum = 0.0f;
        for (float t : times) {
            sq_sum += (t - result.avg_time_ms) * (t - result.avg_time_ms);
        }
        result.std_dev_ms = std::sqrt(sq_sum / times.size());
        
        // Performance metrics
        double flops = 2.0 * M * N * K;
        result.gflops = static_cast<float>(flops / (result.avg_time_ms * 1e6));
        
        // Memory: read A (M*K) + B (K*N), write C (M*N)
        double bytes = (static_cast<double>(M) * K + 
                       static_cast<double>(K) * N + 
                       static_cast<double>(M) * N) * sizeof(float);
        result.memory_bandwidth_gb = static_cast<float>(bytes / (result.avg_time_ms * 1e6));
        
        // Arithmetic intensity: FLOPs / Bytes
        result.arithmetic_intensity = static_cast<float>(flops / bytes);
        
        return result;
    }
    
    // Print profile result
    static void print_result(const ProfileResult& result) {
        printf("┌─────────────────────────────────────────────────────────────────┐\n");
        printf("│ Kernel: %-55s │\n", result.name.c_str());
        printf("├─────────────────────────────────────────────────────────────────┤\n");
        printf("│ Time (ms):     avg=%.4f  min=%.4f  max=%.4f  std=%.4f │\n",
               result.avg_time_ms, result.min_time_ms, result.max_time_ms, result.std_dev_ms);
        printf("│ Performance:   %.2f GFLOPS                                     │\n", result.gflops);
        printf("│ Bandwidth:     %.2f GB/s                                       │\n", result.memory_bandwidth_gb);
        printf("│ Arith. Int.:   %.2f FLOPs/Byte                                 │\n", result.arithmetic_intensity);
        printf("│ Iterations:    %d                                              │\n", result.iterations);
        printf("└─────────────────────────────────────────────────────────────────┘\n");
    }
    
    // Compare multiple results
    static void print_comparison(const std::vector<ProfileResult>& results, 
                                  const std::string& baseline_name = "cuBLAS") {
        // Find baseline
        float baseline_gflops = 0.0f;
        for (const auto& r : results) {
            if (r.name == baseline_name) {
                baseline_gflops = r.gflops;
                break;
            }
        }
        
        printf("\n╔═══════════════════════════════════════════════════════════════════════════════╗\n");
        printf("║                           Performance Comparison                              ║\n");
        printf("╠═══════════════════════════════════════════════════════════════════════════════╣\n");
        printf("║ %-16s │ %10s │ %12s │ %10s │ %10s ║\n", 
               "Kernel", "Time (ms)", "GFLOPS", "Bandwidth", "vs Baseline");
        printf("╠═══════════════════════════════════════════════════════════════════════════════╣\n");
        
        for (const auto& r : results) {
            float ratio = baseline_gflops > 0 ? (r.gflops / baseline_gflops * 100.0f) : 0.0f;
            printf("║ %-16s │ %10.4f │ %12.2f │ %8.1f GB │ %9.1f%% ║\n",
                   r.name.c_str(), r.avg_time_ms, r.gflops, r.memory_bandwidth_gb, ratio);
        }
        
        printf("╚═══════════════════════════════════════════════════════════════════════════════╝\n");
    }
    
private:
    cudaEvent_t start_, stop_;
};

// ============================================================================
// Roofline Model Analysis
// ============================================================================

struct RooflinePoint {
    std::string name;
    float arithmetic_intensity;  // FLOPs/Byte
    float achieved_gflops;
};

class RooflineAnalyzer {
public:
    RooflineAnalyzer(float peak_gflops, float peak_bandwidth_gb) 
        : peak_gflops_(peak_gflops), peak_bandwidth_gb_(peak_bandwidth_gb) {
        // Ridge point: where memory-bound meets compute-bound
        ridge_point_ = peak_gflops_ / peak_bandwidth_gb_;
    }
    
    void add_point(const ProfileResult& result) {
        points_.push_back({result.name, result.arithmetic_intensity, result.gflops});
    }
    
    void analyze() const {
        printf("\n╔═══════════════════════════════════════════════════════════════════════════════╗\n");
        printf("║                           Roofline Model Analysis                             ║\n");
        printf("╠═══════════════════════════════════════════════════════════════════════════════╣\n");
        printf("║ Peak Compute:    %.1f GFLOPS                                                 ║\n", peak_gflops_);
        printf("║ Peak Bandwidth:  %.1f GB/s                                                   ║\n", peak_bandwidth_gb_);
        printf("║ Ridge Point:     %.2f FLOPs/Byte                                             ║\n", ridge_point_);
        printf("╠═══════════════════════════════════════════════════════════════════════════════╣\n");
        
        for (const auto& p : points_) {
            float roofline = std::min(peak_gflops_, p.arithmetic_intensity * peak_bandwidth_gb_);
            float efficiency = p.achieved_gflops / roofline * 100.0f;
            const char* bound = p.arithmetic_intensity < ridge_point_ ? "Memory" : "Compute";
            
            printf("║ %-16s │ AI=%.2f │ %.1f/%.1f GFLOPS │ %.1f%% eff │ %s-bound ║\n",
                   p.name.c_str(), p.arithmetic_intensity, 
                   p.achieved_gflops, roofline, efficiency, bound);
        }
        
        printf("╚═══════════════════════════════════════════════════════════════════════════════╝\n");
    }
    
private:
    float peak_gflops_;
    float peak_bandwidth_gb_;
    float ridge_point_;
    std::vector<RooflinePoint> points_;
};

} // namespace mini_inference
