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
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
endif()

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Compiler flags
# Note: Optimization level is controlled by CMAKE_BUILD_TYPE (Release = -O3).
# Only add CUDA-specific flags here.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# Include directories

# Add tensorcraft-core dependency
if(USE_TENSORCRAFT)
    if(TARGET tensorcraft)
        add_compile_definitions(USE_TENSORCRAFT=1)
        message(STATUS "Using tensorcraft-core from existing target")
    else()
        set(TENSORCRAFT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../02-tensorcraft-core")
        if(EXISTS "${TENSORCRAFT_ROOT}/CMakeLists.txt")
            add_subdirectory("${TENSORCRAFT_ROOT}" tensorcraft-from-parent EXCLUDE_FROM_ALL)
            add_compile_definitions(USE_TENSORCRAFT=1)
            message(STATUS "Using tensorcraft-core from ${TENSORCRAFT_ROOT}")
        else()
            message(WARNING "tensorcraft-core not found at ${TENSORCRAFT_ROOT}, using local GEMM implementations")
            set(USE_TENSORCRAFT OFF)
        endif()
    endif()
endif()

# Source files (explicit list — do not use GLOB_RECURSE)
set(SOURCES
    src/inference_engine.cpp
    src/tensor.cu
    src/naive_matmul.cu
    src/coalesced_gemm.cu
    src/tiled_gemm.cu
    src/double_buffer_gemm.cu
    src/optimized_gemm.cu
    src/fused_gemm.cu
    src/half_gemm.cu
    src/vectorized_gemm.cu
    src/benchmark.cu
)

set(HEADERS
    include/common.h
    include/config.h
    include/tensor.h
    include/kernels.cuh
    include/inference_engine.h
    include/memory_pool.h
    include/stream_manager.h
    include/profiler.h
    include/logger.h
    include/autotuner.h
    include/batch_gemm.h
    include/quantization.h
    include/half_gemm.cuh
    include/vectorized_gemm.cuh
)

# Enable FP16 support
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Main library
add_library(mini_inference STATIC ${SOURCES} ${HEADERS})
target_include_directories(mini_inference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mini_inference CUDA::cudart CUDA::cublas)

# Link tensorcraft if available
if(USE_TENSORCRAFT AND TARGET tensorcraft)
    target_link_libraries(mini_inference tensorcraft)
    if(DEFINED TENSORCRAFT_ROOT)
        target_include_directories(mini_inference PUBLIC ${TENSORCRAFT_ROOT}/include)
    endif()
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
    add_executable(mini_inference_tests ${TEST_SOURCES})
    target_link_libraries(mini_inference_tests mini_inference gtest gtest_main CUDA::cudart CUDA::cublas)

    include(GoogleTest)
    gtest_discover_tests(mini_inference_tests)
endif()
