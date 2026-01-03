#include <gtest/gtest.h>
#include "config.h"
#include <fstream>
#include <cstdlib>

using namespace mini_inference;

class ConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        Config::instance().clear();
    }
    
    void TearDown() override {
        Config::instance().clear();
        // Clean up test files
        std::remove("test_config.ini");
    }
    
    void create_test_config_file(const std::string& content) {
        std::ofstream file("test_config.ini");
        file << content;
        file.close();
    }
};

// ============================================================================
// Basic Value Tests
// ============================================================================

TEST_F(ConfigTest, SetAndGetString) {
    Config::instance().set("key1", "value1");
    EXPECT_EQ(Config::instance().get("key1"), "value1");
}

TEST_F(ConfigTest, GetDefaultString) {
    EXPECT_EQ(Config::instance().get("nonexistent", "default"), "default");
}

TEST_F(ConfigTest, GetEmptyDefault) {
    EXPECT_EQ(Config::instance().get("nonexistent"), "");
}

TEST_F(ConfigTest, SetAndGetInt) {
    Config::instance().set("int_key", "42");
    EXPECT_EQ(Config::instance().get_int("int_key"), 42);
}

TEST_F(ConfigTest, GetIntDefault) {
    EXPECT_EQ(Config::instance().get_int("nonexistent", 100), 100);
}

TEST_F(ConfigTest, GetIntInvalid) {
    Config::instance().set("invalid_int", "not_a_number");
    EXPECT_EQ(Config::instance().get_int("invalid_int", 50), 50);
}

TEST_F(ConfigTest, SetAndGetFloat) {
    Config::instance().set("float_key", "3.14");
    EXPECT_FLOAT_EQ(Config::instance().get_float("float_key"), 3.14f);
}

TEST_F(ConfigTest, GetFloatDefault) {
    EXPECT_FLOAT_EQ(Config::instance().get_float("nonexistent", 2.5f), 2.5f);
}

TEST_F(ConfigTest, GetFloatInvalid) {
    Config::instance().set("invalid_float", "not_a_float");
    EXPECT_FLOAT_EQ(Config::instance().get_float("invalid_float", 1.5f), 1.5f);
}

TEST_F(ConfigTest, SetAndGetBoolTrue) {
    Config::instance().set("bool_true", "true");
    EXPECT_TRUE(Config::instance().get_bool("bool_true"));
    
    Config::instance().set("bool_1", "1");
    EXPECT_TRUE(Config::instance().get_bool("bool_1"));
    
    Config::instance().set("bool_yes", "yes");
    EXPECT_TRUE(Config::instance().get_bool("bool_yes"));
    
    Config::instance().set("bool_on", "on");
    EXPECT_TRUE(Config::instance().get_bool("bool_on"));
}

TEST_F(ConfigTest, SetAndGetBoolFalse) {
    Config::instance().set("bool_false", "false");
    EXPECT_FALSE(Config::instance().get_bool("bool_false"));
    
    Config::instance().set("bool_0", "0");
    EXPECT_FALSE(Config::instance().get_bool("bool_0"));
    
    Config::instance().set("bool_no", "no");
    EXPECT_FALSE(Config::instance().get_bool("bool_no"));
}

TEST_F(ConfigTest, GetBoolDefault) {
    EXPECT_FALSE(Config::instance().get_bool("nonexistent", false));
    EXPECT_TRUE(Config::instance().get_bool("nonexistent", true));
}

// ============================================================================
// Key Management Tests
// ============================================================================

TEST_F(ConfigTest, HasKey) {
    Config::instance().set("existing_key", "value");
    EXPECT_TRUE(Config::instance().has("existing_key"));
    EXPECT_FALSE(Config::instance().has("nonexistent_key"));
}

TEST_F(ConfigTest, GetKeys) {
    Config::instance().set("key1", "value1");
    Config::instance().set("key2", "value2");
    Config::instance().set("key3", "value3");
    
    auto keys = Config::instance().keys();
    EXPECT_EQ(keys.size(), 3);
}

TEST_F(ConfigTest, Clear) {
    Config::instance().set("key1", "value1");
    Config::instance().set("key2", "value2");
    
    Config::instance().clear();
    
    EXPECT_FALSE(Config::instance().has("key1"));
    EXPECT_FALSE(Config::instance().has("key2"));
    EXPECT_EQ(Config::instance().keys().size(), 0);
}

// ============================================================================
// File Loading Tests
// ============================================================================

TEST_F(ConfigTest, LoadFromFile) {
    create_test_config_file(
        "key1=value1\n"
        "key2=value2\n"
        "int_key=42\n"
        "float_key=3.14\n"
        "bool_key=true\n"
    );
    
    EXPECT_TRUE(Config::instance().load_from_file("test_config.ini"));
    
    EXPECT_EQ(Config::instance().get("key1"), "value1");
    EXPECT_EQ(Config::instance().get("key2"), "value2");
    EXPECT_EQ(Config::instance().get_int("int_key"), 42);
    EXPECT_FLOAT_EQ(Config::instance().get_float("float_key"), 3.14f);
    EXPECT_TRUE(Config::instance().get_bool("bool_key"));
}

TEST_F(ConfigTest, LoadFromFileWithComments) {
    create_test_config_file(
        "# This is a comment\n"
        "key1=value1\n"
        "; This is also a comment\n"
        "key2=value2\n"
        "\n"
        "key3=value3\n"
    );
    
    EXPECT_TRUE(Config::instance().load_from_file("test_config.ini"));
    
    EXPECT_EQ(Config::instance().get("key1"), "value1");
    EXPECT_EQ(Config::instance().get("key2"), "value2");
    EXPECT_EQ(Config::instance().get("key3"), "value3");
}

TEST_F(ConfigTest, LoadFromFileWithWhitespace) {
    create_test_config_file(
        "  key1  =  value1  \n"
        "\tkey2\t=\tvalue2\t\n"
    );
    
    EXPECT_TRUE(Config::instance().load_from_file("test_config.ini"));
    
    EXPECT_EQ(Config::instance().get("key1"), "value1");
    EXPECT_EQ(Config::instance().get("key2"), "value2");
}

TEST_F(ConfigTest, LoadFromNonexistentFile) {
    EXPECT_FALSE(Config::instance().load_from_file("nonexistent_file.ini"));
}

TEST_F(ConfigTest, LoadFromFileWithValueContainingEquals) {
    create_test_config_file(
        "equation=a=b+c\n"
    );
    
    EXPECT_TRUE(Config::instance().load_from_file("test_config.ini"));
    EXPECT_EQ(Config::instance().get("equation"), "a=b+c");
}

// ============================================================================
// GEMM Preset Tests
// ============================================================================

TEST_F(ConfigTest, GetGemmPresetSmall) {
    auto preset = get_gemm_preset("small");
    EXPECT_EQ(preset.name, "small");
    EXPECT_EQ(preset.block_m, 64);
    EXPECT_EQ(preset.block_n, 64);
    EXPECT_FALSE(preset.use_double_buffer);
}

TEST_F(ConfigTest, GetGemmPresetMedium) {
    auto preset = get_gemm_preset("medium");
    EXPECT_EQ(preset.name, "medium");
    EXPECT_EQ(preset.block_m, 128);
    EXPECT_EQ(preset.block_n, 128);
    EXPECT_TRUE(preset.use_double_buffer);
}

TEST_F(ConfigTest, GetGemmPresetLarge) {
    auto preset = get_gemm_preset("large");
    EXPECT_EQ(preset.name, "large");
    EXPECT_EQ(preset.block_m, 256);
    EXPECT_TRUE(preset.use_vectorized_load);
}

TEST_F(ConfigTest, GetGemmPresetVolta) {
    auto preset = get_gemm_preset("volta");
    EXPECT_EQ(preset.name, "volta");
}

TEST_F(ConfigTest, GetGemmPresetAmpere) {
    auto preset = get_gemm_preset("ampere");
    EXPECT_EQ(preset.name, "ampere");
}

TEST_F(ConfigTest, GetGemmPresetUnknown) {
    auto preset = get_gemm_preset("unknown");
    EXPECT_EQ(preset.name, "medium");  // Default
}

// ============================================================================
// Device Configuration Tests
// ============================================================================

TEST_F(ConfigTest, GetDeviceConfigDefaults) {
    auto config = get_device_config();
    EXPECT_EQ(config.device_id, 0);
    EXPECT_EQ(config.max_memory_usage, 0);
    EXPECT_TRUE(config.enable_tensor_cores);
    EXPECT_TRUE(config.enable_async_copy);
    EXPECT_EQ(config.num_streams, 1);
}

TEST_F(ConfigTest, GetDeviceConfigFromConfig) {
    Config::instance().set("CUDA_DEVICE", "1");
    Config::instance().set("MAX_GPU_MEMORY_MB", "1024");
    Config::instance().set("ENABLE_TENSOR_CORES", "false");
    Config::instance().set("ENABLE_ASYNC_COPY", "false");
    Config::instance().set("NUM_STREAMS", "4");
    
    auto config = get_device_config();
    EXPECT_EQ(config.device_id, 1);
    EXPECT_EQ(config.max_memory_usage, 1024 * 1024 * 1024);
    EXPECT_FALSE(config.enable_tensor_cores);
    EXPECT_FALSE(config.enable_async_copy);
    EXPECT_EQ(config.num_streams, 4);
}

// ============================================================================
// Singleton Tests
// ============================================================================

TEST_F(ConfigTest, SingletonInstance) {
    Config& instance1 = Config::instance();
    Config& instance2 = Config::instance();
    
    EXPECT_EQ(&instance1, &instance2);
}

TEST_F(ConfigTest, SingletonPersistence) {
    Config::instance().set("persistent_key", "persistent_value");
    
    // Access through another reference
    EXPECT_EQ(Config::instance().get("persistent_key"), "persistent_value");
}
