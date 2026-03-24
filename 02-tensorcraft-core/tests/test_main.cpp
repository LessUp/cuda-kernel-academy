/**
 * @file test_main.cpp
 * @brief Main test runner for TensorCraft-HPC
 */

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <iostream>

int main(int argc, char** argv) {
    // Initialize CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found. Skipping TensorCraft GPU tests." << std::endl;
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Running tests on: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << std::endl;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
