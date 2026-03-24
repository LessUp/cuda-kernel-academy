#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <thread>

#include "logger.h"

namespace {
std::string make_test_path(const char* filename) {
    return (std::filesystem::temp_directory_path() / filename).string();
}
}  // namespace

using namespace mini_inference;

class LoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::instance().set_level(LogLevel::INFO);
        Logger::instance().set_console(false);
        Logger::instance().set_colors(false);
        log_path_ = make_test_path("mini_inference_test.log");
    }

    void TearDown() override {
        std::remove(log_path_.c_str());
        Logger::instance().set_console(true);
    }

    std::string log_path_;
};

// ============================================================================
// Log Level Tests
// ============================================================================

TEST_F(LoggerTest, SetAndGetLevel) {
    Logger::instance().set_level(LogLevel::DEBUG);
    EXPECT_EQ(Logger::instance().get_level(), LogLevel::DEBUG);

    Logger::instance().set_level(LogLevel::ERROR);
    EXPECT_EQ(Logger::instance().get_level(), LogLevel::ERROR);
}

TEST_F(LoggerTest, LogLevelNames) {
    EXPECT_STREQ(log_level_name(LogLevel::TRACE), "TRACE");
    EXPECT_STREQ(log_level_name(LogLevel::DEBUG), "DEBUG");
    EXPECT_STREQ(log_level_name(LogLevel::INFO), "INFO ");
    EXPECT_STREQ(log_level_name(LogLevel::WARN), "WARN ");
    EXPECT_STREQ(log_level_name(LogLevel::ERROR), "ERROR");
    EXPECT_STREQ(log_level_name(LogLevel::FATAL), "FATAL");
}

TEST_F(LoggerTest, LogLevelColors) {
    // Just verify colors are non-empty strings
    EXPECT_NE(strlen(log_level_color(LogLevel::TRACE)), 0);
    EXPECT_NE(strlen(log_level_color(LogLevel::DEBUG)), 0);
    EXPECT_NE(strlen(log_level_color(LogLevel::INFO)), 0);
    EXPECT_NE(strlen(log_level_color(LogLevel::WARN)), 0);
    EXPECT_NE(strlen(log_level_color(LogLevel::ERROR)), 0);
    EXPECT_NE(strlen(log_level_color(LogLevel::FATAL)), 0);
}

// ============================================================================
// File Logging Tests
// ============================================================================

TEST_F(LoggerTest, LogToFile) {
    Logger::instance().set_file(log_path_);
    Logger::instance().set_level(LogLevel::INFO);

    LOG_INFO("Test message");

    // Read the log file
    std::ifstream file(log_path_);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    EXPECT_TRUE(content.find("Test message") != std::string::npos);
    EXPECT_TRUE(content.find("INFO") != std::string::npos);
}

TEST_F(LoggerTest, LogWithFormatting) {
    Logger::instance().set_file(log_path_);
    Logger::instance().set_level(LogLevel::INFO);

    LOG_INFO("Value: %d, Float: %.2f", 42, 3.14);

    std::ifstream file(log_path_);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    EXPECT_TRUE(content.find("Value: 42") != std::string::npos);
    EXPECT_TRUE(content.find("Float: 3.14") != std::string::npos);
}

TEST_F(LoggerTest, LogLevelFiltering) {
    Logger::instance().set_file(log_path_);
    Logger::instance().set_level(LogLevel::WARN);

    LOG_DEBUG("Debug message");
    LOG_INFO("Info message");
    LOG_WARN("Warn message");
    LOG_ERROR("Error message");

    std::ifstream file(log_path_);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    EXPECT_TRUE(content.find("Debug message") == std::string::npos);
    EXPECT_TRUE(content.find("Info message") == std::string::npos);
    EXPECT_TRUE(content.find("Warn message") != std::string::npos);
    EXPECT_TRUE(content.find("Error message") != std::string::npos);
}

TEST_F(LoggerTest, LogOff) {
    Logger::instance().set_file(log_path_);
    Logger::instance().set_level(LogLevel::OFF);

    LOG_TRACE("Trace");
    LOG_DEBUG("Debug");
    LOG_INFO("Info");
    LOG_WARN("Warn");
    LOG_ERROR("Error");
    LOG_FATAL("Fatal");

    std::ifstream file(log_path_);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    EXPECT_TRUE(content.empty());
}

// ============================================================================
// Log Format Tests
// ============================================================================

TEST_F(LoggerTest, LogContainsTimestamp) {
    Logger::instance().set_file(log_path_);
    Logger::instance().set_level(LogLevel::INFO);

    LOG_INFO("Timestamp test");

    std::ifstream file(log_path_);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Check for timestamp pattern: [YYYY-MM-DD HH:MM:SS.mmm]
    std::regex timestamp_pattern(R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\])");
    EXPECT_TRUE(std::regex_search(content, timestamp_pattern));
}

TEST_F(LoggerTest, LogContainsFileAndLine) {
    Logger::instance().set_file(log_path_);
    Logger::instance().set_level(LogLevel::INFO);

    LOG_INFO("File and line test");

    std::ifstream file(log_path_);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Should contain filename (without path) and line number
    EXPECT_TRUE(content.find("test_logger.cpp") != std::string::npos);
    EXPECT_TRUE(content.find(":") != std::string::npos);
}

// ============================================================================
// All Log Levels Tests
// ============================================================================

TEST_F(LoggerTest, AllLogLevels) {
    Logger::instance().set_file(log_path_);
    Logger::instance().set_level(LogLevel::TRACE);

    LOG_TRACE("Trace message");
    LOG_DEBUG("Debug message");
    LOG_INFO("Info message");
    LOG_WARN("Warn message");
    LOG_ERROR("Error message");
    LOG_FATAL("Fatal message");

    std::ifstream file(log_path_);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    EXPECT_TRUE(content.find("TRACE") != std::string::npos);
    EXPECT_TRUE(content.find("DEBUG") != std::string::npos);
    EXPECT_TRUE(content.find("INFO") != std::string::npos);
    EXPECT_TRUE(content.find("WARN") != std::string::npos);
    EXPECT_TRUE(content.find("ERROR") != std::string::npos);
    EXPECT_TRUE(content.find("FATAL") != std::string::npos);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(LoggerTest, ConcurrentLogging) {
    Logger::instance().set_file(log_path_);
    Logger::instance().set_level(LogLevel::INFO);

    const int num_threads = 4;
    const int logs_per_thread = 100;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([t, logs_per_thread]() {
            for (int i = 0; i < logs_per_thread; i++) {
                LOG_INFO("Thread %d, Log %d", t, i);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Count lines in log file
    std::ifstream file(log_path_);
    int line_count = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            line_count++;
        }
    }

    EXPECT_EQ(line_count, num_threads * logs_per_thread);
}

// ============================================================================
// Singleton Tests
// ============================================================================

TEST_F(LoggerTest, SingletonInstance) {
    Logger& instance1 = Logger::instance();
    Logger& instance2 = Logger::instance();

    EXPECT_EQ(&instance1, &instance2);
}

TEST_F(LoggerTest, SingletonPersistence) {
    Logger::instance().set_level(LogLevel::DEBUG);

    // Access through another reference
    EXPECT_EQ(Logger::instance().get_level(), LogLevel::DEBUG);
}

// ============================================================================
// Console Output Tests
// ============================================================================

TEST_F(LoggerTest, ConsoleToggle) {
    // This test just verifies the toggle doesn't crash
    Logger::instance().set_console(true);
    Logger::instance().set_console(false);
    Logger::instance().set_console(true);
}

TEST_F(LoggerTest, ColorToggle) {
    // This test just verifies the toggle doesn't crash
    Logger::instance().set_colors(true);
    Logger::instance().set_colors(false);
    Logger::instance().set_colors(true);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(LoggerTest, EmptyMessage) {
    Logger::instance().set_file(log_path_);
    Logger::instance().set_level(LogLevel::INFO);

    LOG_INFO("");

    std::ifstream file(log_path_);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    EXPECT_TRUE(content.find("INFO") != std::string::npos);
}

TEST_F(LoggerTest, LongMessage) {
    Logger::instance().set_file(log_path_);
    Logger::instance().set_level(LogLevel::INFO);

    std::string long_msg(10000, 'x');
    LOG_INFO("%s", long_msg.c_str());

    std::ifstream file(log_path_);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    EXPECT_TRUE(content.find(long_msg) != std::string::npos);
}

TEST_F(LoggerTest, SpecialCharacters) {
    Logger::instance().set_file(log_path_);
    Logger::instance().set_level(LogLevel::INFO);

    LOG_INFO("Special chars: %%d %%s %%f");

    std::ifstream file(log_path_);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    EXPECT_TRUE(content.find("%d %s %f") != std::string::npos);
}
