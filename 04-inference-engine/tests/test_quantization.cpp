#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "quantization.h"

using namespace mini_inference;

class QuantizationTest : public ::testing::Test {
protected:
    std::vector<float> generate_random_data(size_t n, float min_val = -1.0f, float max_val = 1.0f) {
        std::vector<float> data(n);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (size_t i = 0; i < n; i++) {
            data[i] = dist(gen);
        }
        return data;
    }
};

// ============================================================================
// Basic Quantization Tests
// ============================================================================

TEST_F(QuantizationTest, ComputeQuantParams) {
    std::vector<float> data = {-1.0f, 0.0f, 0.5f, 1.0f};
    auto params = compute_quant_params(data.data(), data.size());

    EXPECT_GT(params.scale, 0.0f);
    EXPECT_EQ(params.zero_point, 0);  // Symmetric quantization
}

TEST_F(QuantizationTest, QuantizeDequantizeRoundTrip) {
    std::vector<float> original = generate_random_data(100);
    auto params = compute_quant_params(original.data(), original.size());

    std::vector<int8_t> quantized(original.size());
    quantize_tensor(original.data(), quantized.data(), original.size(), params);

    std::vector<float> dequantized(original.size());
    dequantize_tensor(quantized.data(), dequantized.data(), original.size(), params);

    // Check that values are close (within quantization error)
    float max_error = 0.0f;
    for (size_t i = 0; i < original.size(); i++) {
        float error = std::abs(original[i] - dequantized[i]);
        max_error = std::max(max_error, error);
    }

    // Error should be bounded by scale
    EXPECT_LT(max_error, params.scale * 1.5f);
}

TEST_F(QuantizationTest, QuantizeClipping) {
    std::vector<float> data = {-200.0f, 0.0f, 200.0f};
    auto params = compute_quant_params(data.data(), data.size());

    std::vector<int8_t> quantized(data.size());
    quantize_tensor(data.data(), quantized.data(), data.size(), params);

    // Extreme values should be clipped to -128/127
    EXPECT_EQ(quantized[0], -127);  // Clipped
    EXPECT_EQ(quantized[2], 127);   // Clipped
}

TEST_F(QuantizationTest, ZeroQuantization) {
    std::vector<float> data = {0.0f, 0.0f, 0.0f};
    auto params = compute_quant_params(data.data(), data.size());

    std::vector<int8_t> quantized(data.size());
    quantize_tensor(data.data(), quantized.data(), data.size(), params);

    for (int8_t val : quantized) {
        EXPECT_EQ(val, 0);
    }
}

// ============================================================================
// QuantizedWeight Tests
// ============================================================================

TEST_F(QuantizationTest, QuantizedWeightConstruction) {
    std::vector<float> weight = generate_random_data(256);
    QuantizedWeight qw(weight.data(), 16, 16);

    EXPECT_EQ(qw.rows, 16);
    EXPECT_EQ(qw.cols, 16);
    EXPECT_EQ(qw.data.size(), 256);
    EXPECT_FLOAT_EQ(qw.compression_ratio(), 4.0f);
}

TEST_F(QuantizationTest, QuantizedWeightDequantize) {
    std::vector<float> original = generate_random_data(64);
    QuantizedWeight qw(original.data(), 8, 8);

    auto dequantized = qw.dequantize();
    EXPECT_EQ(dequantized.size(), 64);

    // Check reconstruction error
    float max_error = 0.0f;
    for (size_t i = 0; i < original.size(); i++) {
        float error = std::abs(original[i] - dequantized[i]);
        max_error = std::max(max_error, error);
    }
    EXPECT_LT(max_error, qw.params.scale * 1.5f);
}

// ============================================================================
// Per-Channel Quantization Tests
// ============================================================================

TEST_F(QuantizationTest, PerChannelQuantParams) {
    // Create data with different ranges per row
    std::vector<float> data = {
        0.1f,   0.2f, 0.3f, 0.4f,  // Row 0: small range
        -1.0f,  0.0f, 0.5f, 1.0f,  // Row 1: medium range
        -10.0f, 5.0f, 0.0f, 10.0f  // Row 2: large range
    };

    auto params = compute_per_channel_params(data.data(), 3, 4);

    EXPECT_EQ(params.num_channels, 3);
    EXPECT_EQ(params.scales.size(), 3);

    // Scales should reflect the range of each row
    EXPECT_LT(params.scales[0], params.scales[1]);
    EXPECT_LT(params.scales[1], params.scales[2]);
}

TEST_F(QuantizationTest, PerChannelRoundTrip) {
    std::vector<float> original = generate_random_data(64);
    int rows = 8, cols = 8;

    auto params = compute_per_channel_params(original.data(), rows, cols);

    std::vector<int8_t> quantized(64);
    quantize_per_channel(original.data(), quantized.data(), rows, cols, params);

    std::vector<float> dequantized(64);
    dequantize_per_channel(quantized.data(), dequantized.data(), rows, cols, params);

    // Check reconstruction error per row
    for (int r = 0; r < rows; r++) {
        float max_error = 0.0f;
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            float error = std::abs(original[idx] - dequantized[idx]);
            max_error = std::max(max_error, error);
        }
        EXPECT_LT(max_error, params.scales[r] * 1.5f);
    }
}

// ============================================================================
// Calibrator Tests
// ============================================================================

TEST_F(QuantizationTest, CalibratorObserve) {
    QuantizationCalibrator calibrator;

    std::vector<float> batch1 = {-1.0f, 0.0f, 1.0f};
    std::vector<float> batch2 = {-2.0f, 0.5f, 2.0f};

    calibrator.observe(batch1.data(), batch1.size());
    calibrator.observe(batch2.data(), batch2.size());

    auto params = calibrator.get_params();

    // Should capture the full range
    EXPECT_GE(params.max_val, 2.0f);
    EXPECT_LE(params.min_val, -2.0f);
}

TEST_F(QuantizationTest, CalibratorStatistics) {
    QuantizationCalibrator calibrator;

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    calibrator.observe(data.data(), data.size());

    EXPECT_FLOAT_EQ(calibrator.mean(), 3.0f);
    EXPECT_FLOAT_EQ(calibrator.variance(), 2.0f);
}

TEST_F(QuantizationTest, CalibratorReset) {
    QuantizationCalibrator calibrator;

    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    calibrator.observe(data.data(), data.size());

    calibrator.reset();

    EXPECT_FLOAT_EQ(calibrator.mean(), 0.0f);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(QuantizationTest, SingleValueQuantization) {
    std::vector<float> data = {0.5f};
    auto params = compute_quant_params(data.data(), data.size());

    std::vector<int8_t> quantized(1);
    quantize_tensor(data.data(), quantized.data(), 1, params);

    std::vector<float> dequantized(1);
    dequantize_tensor(quantized.data(), dequantized.data(), 1, params);

    EXPECT_NEAR(dequantized[0], data[0], params.scale);
}

TEST_F(QuantizationTest, LargeRangeQuantization) {
    std::vector<float> data = {-1000.0f, 0.0f, 1000.0f};
    auto params = compute_quant_params(data.data(), data.size());

    // Scale should handle large range
    EXPECT_GT(params.scale, 7.0f);  // 1000/127 ≈ 7.87
}
