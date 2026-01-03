# TensorCraft-HPC Tests
add_executable(tensorcraft_tests
    test_main.cpp
    test_elementwise.cpp
    test_softmax.cpp
    test_normalization.cpp
    test_gemm.cpp
)

target_link_libraries(tensorcraft_tests PRIVATE
    tensorcraft
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
)

target_include_directories(tensorcraft_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Enable CUDA for test files
set_target_properties(tensorcraft_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
)

# Rename .cpp files to .cu for CUDA compilation
set_source_files_properties(
    test_main.cpp
    test_elementwise.cpp
    test_softmax.cpp
    test_normalization.cpp
    test_gemm.cpp
    PROPERTIES LANGUAGE CUDA
)

include(GoogleTest)
gtest_discover_tests(tensorcraft_tests)
