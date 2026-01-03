#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include "01_elementwise/transpose.cuh"
#include "common/tensor.cuh"
#include "../test_utils.hpp"

// Feature: hpc-ai-optimization-lab, Property 4: Transpose Correctness
RC_GTEST_PROP(TransposeTest, Correctness, ()) {
    auto rows = *rc::gen::inRange<int>(1, 256);
    auto cols = *rc::gen::inRange<int>(1, 256);
    auto input = *rc::gen::container<std::vector<float>>(rows * cols, rc::gen::arbitrary<float>());
    
    // CPU reference
    std::vector<float> expected(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            expected[j * rows + i] = input[i * cols + j];
        }
    }
    
    // GPU implementation
    hpc::Tensor<float> d_input(rows * cols);
    hpc::Tensor<float> d_output(rows * cols);
    d_input.copy_from_host(input);
    
    hpc::elementwise::transpose<float, hpc::elementwise::TransposeOpt::SharedMemPadded>(
        d_input.data(), d_output.data(), rows, cols);
    cudaDeviceSynchronize();
    
    auto result = d_output.to_host();
    
    for (size_t i = 0; i < expected.size(); ++i) {
        RC_ASSERT(hpc::test::almost_equal(result[i], expected[i]));
    }
}

// Feature: hpc-ai-optimization-lab, Property 5: Transpose Involution
RC_GTEST_PROP(TransposeTest, Involution, ()) {
    auto rows = *rc::gen::inRange<int>(1, 128);
    auto cols = *rc::gen::inRange<int>(1, 128);
    auto input = *rc::gen::container<std::vector<float>>(rows * cols, rc::gen::arbitrary<float>());
    
    hpc::Tensor<float> d_input(rows * cols);
    hpc::Tensor<float> d_temp(rows * cols);
    hpc::Tensor<float> d_output(rows * cols);
    d_input.copy_from_host(input);
    
    // Transpose twice
    hpc::elementwise::transpose<float, hpc::elementwise::TransposeOpt::SharedMemPadded>(
        d_input.data(), d_temp.data(), rows, cols);
    hpc::elementwise::transpose<float, hpc::elementwise::TransposeOpt::SharedMemPadded>(
        d_temp.data(), d_output.data(), cols, rows);
    cudaDeviceSynchronize();
    
    auto result = d_output.to_host();
    
    for (size_t i = 0; i < input.size(); ++i) {
        RC_ASSERT(hpc::test::almost_equal(result[i], input[i]));
    }
}
