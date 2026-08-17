# Standalone convenience build for the SGEMM tutorial.
#
# The canonical build/test path for all modules is the repository root CMake
# preset (`cmake --preset default && ctest --preset default`).  This Makefile is
# kept for quick benchmark runs and for readers who prefer a single nvcc line.

NVCC ?= $(shell command -v nvcc)
GPU_ARCH ?= sm_86
CXXFLAGS := -std=c++17 -O3 --use_fast_math -lineinfo -arch=$(GPU_ARCH)
INCLUDES := -I./src
LIBS := -lcublas -lcurand

BUILD_DIR := build
BENCHMARK_BIN := $(BUILD_DIR)/sgemm_benchmark
TEST_BIN := $(BUILD_DIR)/test_sgemm

# Self-contained GoogleTest download so `make test` does not depend on a
# distro gtest package.
GTEST_VERSION := v1.14.0
GTEST_DIR := $(BUILD_DIR)/googletest
GTEST_INCLUDE := $(GTEST_DIR)/googletest/include
GTEST_MAIN_INCLUDE := $(GTEST_DIR)/googletest
# test_sgemm.cu defines its own main(), so only gtest-all is needed.
GTEST_SRCS := $(GTEST_DIR)/googletest/src/gtest-all.cc
TEST_LIBS := -lpthread

.PHONY: all benchmark test clean

all: benchmark

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

benchmark: $(BENCHMARK_BIN)

$(BENCHMARK_BIN): src/main.cu | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

$(GTEST_DIR)/googletest/src/gtest-all.cc:
	mkdir -p $(BUILD_DIR)
	git clone --depth 1 --branch $(GTEST_VERSION) https://github.com/google/googletest.git $(GTEST_DIR)

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): tests/test_sgemm.cu $(GTEST_SRCS) | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $< $(GTEST_SRCS) \
		-I$(GTEST_INCLUDE) -I$(GTEST_MAIN_INCLUDE) \
		-o $@ $(LIBS) $(TEST_LIBS)

clean:
	rm -rf $(BUILD_DIR)
