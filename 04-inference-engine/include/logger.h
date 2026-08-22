#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

namespace mini_inference {

// ============================================================================
// Logging System
// Thread-safe logging with multiple output targets and log levels
// ============================================================================

enum class LogLevel { TRACE = 0, DEBUG = 1, INFO = 2, WARN = 3, ERROR = 4, FATAL = 5, OFF = 6 };

inline const char* log_level_name(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE:
            return "TRACE";
        case LogLevel::DEBUG:
            return "DEBUG";
        case LogLevel::INFO:
            return "INFO ";
        case LogLevel::WARN:
            return "WARN ";
        case LogLevel::ERROR:
            return "ERROR";
        case LogLevel::FATAL:
            return "FATAL";
        default:
            return "UNKNOWN";
    }
}

inline const char* log_level_color(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE:
            return "\033[90m";  // Gray
        case LogLevel::DEBUG:
            return "\033[36m";  // Cyan
        case LogLevel::INFO:
            return "\033[32m";  // Green
        case LogLevel::WARN:
            return "\033[33m";  // Yellow
        case LogLevel::ERROR:
            return "\033[31m";  // Red
        case LogLevel::FATAL:
            return "\033[35m";  // Magenta
        default:
            return "\033[0m";
    }
}

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void set_level(LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        level_ = level;
    }

    LogLevel get_level() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return level_;
    }

    void set_file(const std::string& path) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) {
            file_.close();
        }
        file_.open(path, std::ios::out | std::ios::app);
        log_to_file_ = file_.is_open();
    }

    void set_console(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        log_to_console_ = enabled;
    }

    void set_colors(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        use_colors_ = enabled;
    }

    template <typename... Args>
    void log(LogLevel level, const char* file, int line, const char* fmt, Args... args) {
        if (level < level_)
            return;

        std::lock_guard<std::mutex> lock(mutex_);

        // Format timestamp
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        std::ostringstream timestamp;
        timestamp << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << '.'
                  << std::setfill('0') << std::setw(3) << ms.count();

        // Format message
        std::string message = format_string(fmt, args...);

        // Extract filename from path
        std::string filename = file;
        auto pos = filename.find_last_of("/\\");
        if (pos != std::string::npos) {
            filename = filename.substr(pos + 1);
        }

        // Output to console
        if (log_to_console_) {
            if (use_colors_) {
                std::cout << log_level_color(level);
            }
            std::cout << "[" << timestamp.str() << "] "
                      << "[" << log_level_name(level) << "] "
                      << "[" << filename << ":" << line << "] " << message;
            if (use_colors_) {
                std::cout << "\033[0m";
            }
            std::cout << std::endl;
        }

        // Output to file
        if (log_to_file_ && file_.is_open()) {
            file_ << "[" << timestamp.str() << "] "
                  << "[" << log_level_name(level) << "] "
                  << "[" << filename << ":" << line << "] " << message << std::endl;
        }
    }

private:
    Logger() = default;
    ~Logger() {
        if (file_.is_open()) {
            file_.close();
        }
    }

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    template <typename... Args>
    std::string format_string(const char* fmt, Args... args) {
        int size = std::snprintf(nullptr, 0, fmt, args...) + 1;
        if (size <= 0)
            return fmt;
        std::unique_ptr<char[]> buf(new char[size]);
        std::snprintf(buf.get(), size, fmt, args...);
        return std::string(buf.get(), buf.get() + size - 1);
    }

    // No-argument overload: return the string verbatim.  Passing a runtime
    // string straight into snprintf as a format string would misparse any
    // embedded '%' characters (and trips -Wformat-security).
    std::string format_string(const char* fmt) { return std::string(fmt); }

    LogLevel level_ = LogLevel::INFO;
    bool log_to_console_ = true;
    bool log_to_file_ = false;
    bool use_colors_ = true;
    std::ofstream file_;
    mutable std::mutex mutex_;
};

// Logging macros
#define LOG_TRACE(...)                                                                          \
    mini_inference::Logger::instance().log(mini_inference::LogLevel::TRACE, __FILE__, __LINE__, \
                                           __VA_ARGS__)
#define LOG_DEBUG(...)                                                                          \
    mini_inference::Logger::instance().log(mini_inference::LogLevel::DEBUG, __FILE__, __LINE__, \
                                           __VA_ARGS__)
#define LOG_INFO(...)                                                                          \
    mini_inference::Logger::instance().log(mini_inference::LogLevel::INFO, __FILE__, __LINE__, \
                                           __VA_ARGS__)
#define LOG_WARN(...)                                                                          \
    mini_inference::Logger::instance().log(mini_inference::LogLevel::WARN, __FILE__, __LINE__, \
                                           __VA_ARGS__)
#define LOG_ERROR(...)                                                                          \
    mini_inference::Logger::instance().log(mini_inference::LogLevel::ERROR, __FILE__, __LINE__, \
                                           __VA_ARGS__)
#define LOG_FATAL(...)                                                                          \
    mini_inference::Logger::instance().log(mini_inference::LogLevel::FATAL, __FILE__, __LINE__, \
                                           __VA_ARGS__)

}  // namespace mini_inference
