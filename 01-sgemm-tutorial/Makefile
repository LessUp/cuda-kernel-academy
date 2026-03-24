CUDA_HOME ?= /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc
GPU_ARCH ?= sm_86
CXXFLAGS := -std=c++17 -O3 --use_fast_math -lineinfo -arch=$(GPU_ARCH)
INCLUDES := -I./src
LIBS := -lcublas -lcurand
TEST_LIBS := -lgtest -lgtest_main -lpthread
BUILD_DIR := build
BENCHMARK_BIN := $(BUILD_DIR)/sgemm_benchmark
TEST_BIN := $(BUILD_DIR)/test_sgemm

.PHONY: all benchmark test clean

all: benchmark

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

benchmark: $(BENCHMARK_BIN)

$(BENCHMARK_BIN): src/main.cu | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): tests/test_sgemm.cu | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS) $(TEST_LIBS)

clean:
	rm -rf $(BUILD_DIR)
