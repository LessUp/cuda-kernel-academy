#pragma once

#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mini_inference {

// ============================================================================
// Configuration Management
// Load and manage configuration from files or environment
// ============================================================================

class Config {
public:
    static Config& instance() {
        static Config config;
        return config;
    }
    
    // Load configuration from file
    bool load_from_file(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#' || line[0] == ';') {
                continue;
            }
            
            auto pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = trim(line.substr(0, pos));
                std::string value = trim(line.substr(pos + 1));
                values_[key] = value;
            }
        }
        
        return true;
    }
    
    // Get string value
    std::string get(const std::string& key, const std::string& default_value = "") const {
        auto it = values_.find(key);
        if (it != values_.end()) {
            return it->second;
        }
        
        // Check environment variable
        const char* env = std::getenv(key.c_str());
        if (env) {
            return env;
        }
        
        return default_value;
    }
    
    // Get integer value
    int get_int(const std::string& key, int default_value = 0) const {
        std::string value = get(key);
        if (value.empty()) {
            return default_value;
        }
        try {
            return std::stoi(value);
        } catch (...) {
            return default_value;
        }
    }
    
    // Get float value
    float get_float(const std::string& key, float default_value = 0.0f) const {
        std::string value = get(key);
        if (value.empty()) {
            return default_value;
        }
        try {
            return std::stof(value);
        } catch (...) {
            return default_value;
        }
    }
    
    // Get boolean value
    bool get_bool(const std::string& key, bool default_value = false) const {
        std::string value = get(key);
        if (value.empty()) {
            return default_value;
        }
        return value == "true" || value == "1" || value == "yes" || value == "on";
    }
    
    // Set value
    void set(const std::string& key, const std::string& value) {
        values_[key] = value;
    }
    
    // Check if key exists
    bool has(const std::string& key) const {
        return values_.find(key) != values_.end();
    }
    
    // Clear all values
    void clear() {
        values_.clear();
    }
    
    // Get all keys
    std::vector<std::string> keys() const {
        std::vector<std::string> result;
        for (const auto& [key, value] : values_) {
            result.push_back(key);
        }
        return result;
    }
    
private:
    Config() = default;
    
    static std::string trim(const std::string& str) {
        size_t start = str.find_first_not_of(" \t\r\n");
        size_t end = str.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        return str.substr(start, end - start + 1);
    }
    
    std::map<std::string, std::string> values_;
};

// ============================================================================
// GEMM Configuration Presets
// ============================================================================

struct GemmPreset {
    std::string name;
    int block_m;
    int block_n;
    int block_k;
    int thread_m;
    int thread_n;
    bool use_double_buffer;
    bool use_vectorized_load;
};

inline GemmPreset get_gemm_preset(const std::string& name) {
    static std::map<std::string, GemmPreset> presets = {
        {"small", {"small", 64, 64, 8, 4, 4, false, false}},
        {"medium", {"medium", 128, 128, 8, 8, 8, true, true}},
        {"large", {"large", 256, 128, 16, 8, 8, true, true}},
        {"volta", {"volta", 128, 128, 8, 8, 8, true, true}},
        {"ampere", {"ampere", 128, 256, 16, 8, 8, true, true}},
    };
    
    auto it = presets.find(name);
    if (it != presets.end()) {
        return it->second;
    }
    return presets["medium"];  // Default
}

// ============================================================================
// Device Configuration
// ============================================================================

struct DeviceConfig {
    int device_id = 0;
    size_t max_memory_usage = 0;  // 0 = unlimited
    bool enable_tensor_cores = true;
    bool enable_async_copy = true;
    int num_streams = 1;
};

inline DeviceConfig get_device_config() {
    DeviceConfig config;
    config.device_id = Config::instance().get_int("CUDA_DEVICE", 0);
    config.max_memory_usage = Config::instance().get_int("MAX_GPU_MEMORY_MB", 0) * 1024 * 1024;
    config.enable_tensor_cores = Config::instance().get_bool("ENABLE_TENSOR_CORES", true);
    config.enable_async_copy = Config::instance().get_bool("ENABLE_ASYNC_COPY", true);
    config.num_streams = Config::instance().get_int("NUM_STREAMS", 1);
    return config;
}

} // namespace mini_inference
