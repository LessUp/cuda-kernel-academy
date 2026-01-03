cmake_minimum_required(VERSION 3.20)
project(TensorCraft-HPC 
    VERSION 1.0.0
    DESCRIPTION "High-Performance AI Kernels with Modern C++ & CUDA"
    LANGUAGES CXX CUDA
)

# ============================================================================
# C++ Standard Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Detect C++20/23 support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++20" COMPILER_SUPPORTS_CXX20)
check_cxx_compiler_flag("-std=c++23" COMPILER_SUPPORTS_CXX23)

if(COMPILER_SUPPORTS_CXX23)
    set(CMAKE_CXX_STANDARD 23)
    add_compile_definitions(TC_CPP23=1 TC_CPP20=1 TC_CPP17=1)
    message(STATUS "C++23 support enabled")
elseif(COMPILER_SUPPORTS_CXX20)
    set(CMAKE_CXX_STANDARD 20)
    add_compile_definitions(TC_CPP20=1 TC_CPP17=1)
    message(STATUS "C++20 support enabled")
else()
    add_compile_definitions(TC_CPP17=1)
    message(STATUS "C++17 support enabled")
endif()

# ============================================================================
# CUDA Configuration
# ============================================================================
find_package(CUDAToolkit REQUIRED)

# Check CUDA version
if(CUDAToolkit_VERSION VERSION_LESS "11.0")
    message(FATAL_ERROR "CUDA 11.0 or higher is required. Found: ${CUDAToolkit_VERSION}")
endif()

message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures - support from Volta to Blackwell
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures")

# Feature detection based on CUDA version
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    add_compile_definitions(TC_CUDA_13=1 TC_CUDA_12=1 TC_CUDA_11=1)
    add_compile_definitions(TC_HAS_TMA=1 TC_HAS_WGMMA=1 TC_HAS_FP8=1 TC_HAS_WMMA=1)
    message(STATUS "CUDA 13.x features enabled (TMA, WGMMA, FP8, WMMA)")
elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
    add_compile_definitions(TC_CUDA_12=1 TC_CUDA_11=1)
    add_compile_definitions(TC_HAS_TMA=1 TC_HAS_WGMMA=1 TC_HAS_FP8=1 TC_HAS_WMMA=1)
    message(STATUS "CUDA 12.x features enabled (TMA, WGMMA, FP8, WMMA)")
elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
    add_compile_definitions(TC_CUDA_11=1)
    add_compile_definitions(TC_HAS_WMMA=1)
    message(STATUS "CUDA 11.x features enabled (WMMA)")
endif()

# ============================================================================
# Build Options
# ============================================================================
option(TC_BUILD_TESTS "Build unit tests" ON)
option(TC_BUILD_BENCHMARKS "Build benchmarks" ON)
option(TC_BUILD_PYTHON "Build Python bindings" ON)
option(TC_BUILD_DOCS "Build documentation" OFF)

# ============================================================================
# Dependencies via FetchContent
# ============================================================================
include(FetchContent)

if(TC_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

if(TC_BUILD_BENCHMARKS)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

if(TC_BUILD_PYTHON)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ============================================================================
# Main Library Target (Header-Only)
# ============================================================================
add_library(tensorcraft INTERFACE)
target_include_directories(tensorcraft INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tensorcraft INTERFACE CUDA::cudart)

# Add alias for consistent naming
add_library(TensorCraft::tensorcraft ALIAS tensorcraft)

# ============================================================================
# Subdirectories
# ============================================================================
if(TC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TC_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(TC_BUILD_PYTHON)
    add_subdirectory(src/python_ops)
endif()

# ============================================================================
# Installation
# ============================================================================
include(GNUInstallDirs)

install(TARGETS tensorcraft
    EXPORT TensorCraftTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT TensorCraftTargets
    FILE TensorCraftTargets.cmake
    NAMESPACE TensorCraft::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorCraft
)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "TensorCraft-HPC Configuration Summary:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Version:   ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA Archs:     ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Tests:    ${TC_BUILD_TESTS}")
message(STATUS "  Build Bench:    ${TC_BUILD_BENCHMARKS}")
message(STATUS "  Build Python:   ${TC_BUILD_PYTHON}")
message(STATUS "")
