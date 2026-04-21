#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "common.h"

namespace mini_inference {

// ============================================================================
// Quantization Support
// INT8 quantization for inference acceleration
// ============================================================================

struct QuantizationParams {
    float scale = 1.0f;
    int32_t zero_point = 0;
    float min_val = -128.0f;
    float max_val = 127.0f;
};

// Compute quantization parameters from tensor statistics
inline QuantizationParams compute_quant_params(const float* data, size_t n) {
    QuantizationParams params;

    // Find min/max
    float min_val = data[0];
    float max_val = data[0];
    for (size_t i = 1; i < n; i++) {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }

    // Symmetric quantization
    float abs_max = std::max(std::abs(min_val), std::abs(max_val));
    params.scale = abs_max / 127.0f;
    params.zero_point = 0;
    params.min_val = -abs_max;
    params.max_val = abs_max;

    // Avoid division by zero
    if (params.scale < 1e-8f) {
        params.scale = 1e-8f;
    }

    return params;
}

// Quantize float tensor to int8
inline void quantize_tensor(const float* input, int8_t* output, size_t n,
                            const QuantizationParams& params) {
    float inv_scale = 1.0f / params.scale;
    for (size_t i = 0; i < n; i++) {
        float val = input[i] * inv_scale;
        val = std::max(-128.0f, std::min(127.0f, std::round(val)));
        output[i] = static_cast<int8_t>(val);
    }
}

// Dequantize int8 tensor to float
inline void dequantize_tensor(const int8_t* input, float* output, size_t n,
                              const QuantizationParams& params) {
    for (size_t i = 0; i < n; i++) {
        output[i] = static_cast<float>(input[i]) * params.scale;
    }
}

// ============================================================================
// Quantized Weight Storage
// ============================================================================

struct QuantizedWeight {
    std::vector<int8_t> data;
    QuantizationParams params;
    int rows = 0;
    int cols = 0;

    QuantizedWeight() = default;

    QuantizedWeight(const float* weight, int r, int c) : rows(r), cols(c) {
        size_t n = static_cast<size_t>(r) * c;
        data.resize(n);
        params = compute_quant_params(weight, n);
        quantize_tensor(weight, data.data(), n, params);
    }

    // Dequantize to float
    std::vector<float> dequantize() const {
        std::vector<float> result(data.size());
        dequantize_tensor(data.data(), result.data(), data.size(), params);
        return result;
    }

    // Get memory savings ratio
    float compression_ratio() const {
        return 4.0f;  // float32 -> int8 = 4x compression
    }
};

// ============================================================================
// Per-Channel Quantization (better accuracy for weights)
// ============================================================================

struct PerChannelQuantParams {
    std::vector<float> scales;
    std::vector<int32_t> zero_points;
    int num_channels = 0;
};

inline PerChannelQuantParams compute_per_channel_params(const float* data, int rows, int cols) {
    PerChannelQuantParams params;
    params.num_channels = rows;
    params.scales.resize(rows);
    params.zero_points.resize(rows, 0);

    for (int r = 0; r < rows; r++) {
        float min_val = data[r * cols];
        float max_val = data[r * cols];
        for (int c = 1; c < cols; c++) {
            float val = data[r * cols + c];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }

        float abs_max = std::max(std::abs(min_val), std::abs(max_val));
        params.scales[r] = abs_max / 127.0f;
        if (params.scales[r] < 1e-8f) {
            params.scales[r] = 1e-8f;
        }
    }

    return params;
}

inline void quantize_per_channel(const float* input, int8_t* output, int rows, int cols,
                                 const PerChannelQuantParams& params) {
    for (int r = 0; r < rows; r++) {
        float inv_scale = 1.0f / params.scales[r];
        for (int c = 0; c < cols; c++) {
            float val = input[r * cols + c] * inv_scale;
            val = std::max(-128.0f, std::min(127.0f, std::round(val)));
            output[r * cols + c] = static_cast<int8_t>(val);
        }
    }
}

inline void dequantize_per_channel(const int8_t* input, float* output, int rows, int cols,
                                   const PerChannelQuantParams& params) {
    for (int r = 0; r < rows; r++) {
        float scale = params.scales[r];
        for (int c = 0; c < cols; c++) {
            output[r * cols + c] = static_cast<float>(input[r * cols + c]) * scale;
        }
    }
}

// ============================================================================
// Calibration for Dynamic Quantization
// ============================================================================

class QuantizationCalibrator {
public:
    void observe(const float* data, size_t n) {
        for (size_t i = 0; i < n; i++) {
            min_observed_ = std::min(min_observed_, data[i]);
            max_observed_ = std::max(max_observed_, data[i]);
            sum_ += data[i];
            sum_sq_ += data[i] * data[i];
            count_++;
        }
    }

    QuantizationParams get_params() const {
        QuantizationParams params;

        // Use observed range
        float abs_max = std::max(std::abs(min_observed_), std::abs(max_observed_));
        params.scale = abs_max / 127.0f;
        params.zero_point = 0;
        params.min_val = -abs_max;
        params.max_val = abs_max;

        if (params.scale < 1e-8f) {
            params.scale = 1e-8f;
        }

        return params;
    }

    void reset() {
        min_observed_ = std::numeric_limits<float>::max();
        max_observed_ = std::numeric_limits<float>::lowest();
        sum_ = 0.0;
        sum_sq_ = 0.0;
        count_ = 0;
    }

    float mean() const { return count_ > 0 ? sum_ / count_ : 0.0f; }
    float variance() const {
        if (count_ == 0)
            return 0.0f;
        double mean_val = sum_ / count_;
        return static_cast<float>(sum_sq_ / count_ - mean_val * mean_val);
    }

private:
    float min_observed_ = std::numeric_limits<float>::max();
    float max_observed_ = std::numeric_limits<float>::lowest();
    double sum_ = 0.0;
    double sum_sq_ = 0.0;
    size_t count_ = 0;
};

}  // namespace mini_inference
