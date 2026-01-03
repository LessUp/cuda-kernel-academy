cmake_minimum_required(VERSION 3.20)
project(mini-inference-engine LANGUAGES CXX CUDA)

# Options
option(BUILD_TESTS "Build test suite" ON)
option(USE_TENSORCRAFT "Use tensorcraft-core for GEMM operations" ON)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture (adjust for your GPU)
# Supported: 70, 75, 80, 86, 87, 88, 89, 90
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add tensorcraft-core dependency
if(USE_TENSORCRAFT)
    set(TENSORCRAFT_ROOT "${CMAKE_SOURCE_DIR}/../02-tensorcraft-core")
    if(EXISTS ${TENSORCRAFT_ROOT})
        add_subdirectory(${TENSORCRAFT_ROOT} tensorcraft EXCLUDE_FROM_ALL)
        add_compile_definitions(USE_TENSORCRAFT=1)
        message(STATUS "Using tensorcraft-core from ${TENSORCRAFT_ROOT}")
    else()
        message(WARNING "tensorcraft-core not found at ${TENSORCRAFT_ROOT}, using local GEMM implementations")
        set(USE_TENSORCRAFT OFF)
    endif()
endif()

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")
file(GLOB_RECURSE HEADERS "include/*.h" "include/*.cuh")

# Enable FP16 support
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Main library
add_library(mini_inference STATIC ${SOURCES} ${HEADERS})
target_link_libraries(mini_inference CUDA::cudart CUDA::cublas)

# Link tensorcraft if available
if(USE_TENSORCRAFT AND TARGET tensorcraft)
    target_link_libraries(mini_inference tensorcraft)
    target_include_directories(mini_inference PUBLIC ${TENSORCRAFT_ROOT}/include)
endif()

# Benchmark executable
add_executable(benchmark benchmarks/benchmark.cpp)
target_link_libraries(benchmark mini_inference CUDA::cudart CUDA::cublas)

# MNIST inference demo
add_executable(mnist_demo benchmarks/mnist_demo.cpp)
target_link_libraries(mnist_demo mini_inference CUDA::cudart CUDA::cublas)

# Detailed benchmark
add_executable(detailed_benchmark benchmarks/detailed_benchmark.cu)
target_link_libraries(detailed_benchmark mini_inference CUDA::cudart CUDA::cublas)

# Testing with Google Test (optional)
if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
        URL_HASH MD5=c8340a482851ef6a3fe618a082304cfc
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp" "tests/*.cu")
    add_executable(tests ${TEST_SOURCES})
    target_link_libraries(tests mini_inference gtest gtest_main CUDA::cudart CUDA::cublas)

    include(GoogleTest)
    gtest_discover_tests(tests)
endif()
