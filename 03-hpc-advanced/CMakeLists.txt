cmake_minimum_required(VERSION 3.24)
project(HPC-AI-Optimization-Lab LANGUAGES CXX CUDA)

# C++20 and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA settings
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Auto-detect GPU architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" INSTALLED_GPU_CCS_4 "${INSTALLED_GPU_CCS_3}")
    set(CMAKE_CUDA_ARCHITECTURES ${INSTALLED_GPU_CCS_4})
    message(STATUS "Auto-detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# FetchContent for dependencies
include(FetchContent)

# GoogleTest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# fmt library
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.2.1
)

# RapidCheck for property-based testing
FetchContent_Declare(
    rapidcheck
    GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
    GIT_TAG master
)

FetchContent_MakeAvailable(googletest fmt rapidcheck)

# CUTLASS (header-only)
FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG v3.5.0
)
FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
    FetchContent_Populate(cutlass)
endif()

# Nanobind (optional, for Python bindings)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
if(BUILD_PYTHON_BINDINGS)
    find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v2.0.0
    )
    FetchContent_MakeAvailable(nanobind)
endif()


# Common library (header-only)
add_library(hpc_common INTERFACE)
target_include_directories(hpc_common INTERFACE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/common
    ${cutlass_SOURCE_DIR}/include
)
target_link_libraries(hpc_common INTERFACE fmt::fmt)

# CUDA compile options
function(hpc_add_cuda_library target)
    add_library(${target} ${ARGN})
    target_link_libraries(${target} PUBLIC hpc_common)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            --extended-lambda
            -Xcompiler=-Wall
        >
    )
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endfunction()

# Kernel modules
add_subdirectory(src/01_elementwise)
add_subdirectory(src/02_reduction)
add_subdirectory(src/03_gemm)
add_subdirectory(src/04_convolution)
add_subdirectory(src/05_attention)
add_subdirectory(src/06_quantization)
add_subdirectory(src/07_cuda13_features)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

# Tests
enable_testing()
add_subdirectory(tests)

# Summary
message(STATUS "")
message(STATUS "=== HPC-AI-Optimization-Lab Configuration ===")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Python Bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "==============================================")
