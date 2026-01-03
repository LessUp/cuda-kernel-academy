# Tests configuration
include(GoogleTest)

# Test utilities library
add_library(test_utils INTERFACE)
target_include_directories(test_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_utils INTERFACE 
    GTest::gtest_main 
    rapidcheck
    hpc_common
)

# Helper function to add tests
function(hpc_add_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE test_utils)
    gtest_discover_tests(${target})
endfunction()

# Add test subdirectories
add_subdirectory(common)
add_subdirectory(elementwise)
add_subdirectory(reduction)
add_subdirectory(gemm)
add_subdirectory(attention)
add_subdirectory(cuda13)
add_subdirectory(quantization)
add_subdirectory(convolution)
