# =============================================================================
# CUDA Kernel Academy - Root CMakeLists.txt
# =============================================================================
# A comprehensive learning project for high-performance CUDA kernel development
# =============================================================================

cmake_minimum_required(VERSION 3.20)

# Read version from VERSION file
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" VERSION_STRING)
string(STRIP "${VERSION_STRING}" VERSION_STRING)

project(
  CUDAKernelAcademy
  VERSION ${VERSION_STRING}
  LANGUAGES CXX CUDA
  DESCRIPTION "Comprehensive CUDA kernel optimization learning project"
  HOMEPAGE_URL "https://github.com/LessUp/cuda-kernel-academy")

# =============================================================================
# Build Options
# =============================================================================

option(BUILD_TENSORCRAFT "Build 02-tensorcraft-core module" ON)
option(BUILD_HPC_ADVANCED "Build 03-hpc-advanced module" ON)
option(BUILD_INFERENCE_ENGINE "Build 04-inference-engine module" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

# =============================================================================
# C++ Standard Configuration
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# =============================================================================
# CUDA Configuration
# =============================================================================

# Default CUDA architectures (Volta to Hopper)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES
      "70;75;80;86"
      CACHE STRING "CUDA architectures")
endif()

# CUDA compile options
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

# =============================================================================
# Compiler Warnings
# =============================================================================

add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>)

# =============================================================================
# Build Type Configuration
# =============================================================================

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG")

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")

# =============================================================================
# Output Directories
# =============================================================================

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# =============================================================================
# Common Include Directory
# =============================================================================

add_subdirectory(common)

# =============================================================================
# Subprojects
# =============================================================================

# Note: 01-sgemm-tutorial is a standalone tutorial module with its own Makefile.
# Build it separately with: cd 01-sgemm-tutorial && make GPU_ARCH=sm_86

if(BUILD_TENSORCRAFT)
  message(STATUS "Building 02-tensorcraft-core")
  set(TC_BUILD_TESTS
      ${BUILD_TESTS}
      CACHE BOOL "Build unit tests" FORCE)
  set(TC_BUILD_BENCHMARKS
      ${BUILD_BENCHMARKS}
      CACHE BOOL "Build benchmarks" FORCE)
  set(TC_BUILD_PYTHON
      ${BUILD_PYTHON_BINDINGS}
      CACHE BOOL "Build Python bindings" FORCE)
  add_subdirectory(02-tensorcraft-core)
endif()

if(BUILD_HPC_ADVANCED)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.0")
    message(
      WARNING
        "Skipping 03-hpc-advanced (requires CUDA >= 12.0). Configure with -DBUILD_HPC_ADVANCED=OFF to disable this warning."
    )
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.0")
    message(
      WARNING
        "Skipping 03-hpc-advanced (requires GCC >= 11 for C++20 concepts support). Configure with -DBUILD_HPC_ADVANCED=OFF to disable this warning."
    )
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0")
    message(
      WARNING
        "Skipping 03-hpc-advanced (requires Clang >= 14 for C++20 concepts support). Configure with -DBUILD_HPC_ADVANCED=OFF to disable this warning."
    )
  else()
    message(STATUS "Building 03-hpc-advanced")
    add_subdirectory(03-hpc-advanced)
  endif()
endif()

if(BUILD_INFERENCE_ENGINE)
  if(NOT BUILD_TENSORCRAFT)
    message(
      FATAL_ERROR
        "04-inference-engine requires 02-tensorcraft-core. Enable BUILD_TENSORCRAFT."
    )
  endif()
  message(STATUS "Building 04-inference-engine")
  add_subdirectory(04-inference-engine)
endif()

if(BUILD_EXAMPLES)
  message(STATUS "Building examples")
  add_subdirectory(examples)
endif()

# =============================================================================
# Testing
# =============================================================================

if(BUILD_TESTS)
  enable_testing()
  include(CTest)
endif()

# =============================================================================
# Installation
# =============================================================================

include(GNUInstallDirs)

# Install common headers
install(
  DIRECTORY common/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.cuh"
  PATTERN "*.h")

# Install documentation
install(FILES README.md LICENSE CHANGELOG.md CONTRIBUTING.md CODE_OF_CONDUCT.md
        DESTINATION ${CMAKE_INSTALL_DOCDIR})

# =============================================================================
# Package Configuration
# =============================================================================

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/CUDAKernelAcademyConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "=== CUDA Kernel Academy Configuration ===")
message(STATUS "Version:              ${PROJECT_VERSION}")
message(STATUS "Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures:   ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ standard:         ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA standard:        ${CMAKE_CUDA_STANDARD}")
message(STATUS "")
message(STATUS "Modules:")
message(
  STATUS
    "  01-sgemm-tutorial:    Standalone Makefile (cd 01-sgemm-tutorial && make GPU_ARCH=sm_86)"
)
message(STATUS "  02-tensorcraft-core:  ${BUILD_TENSORCRAFT}")
message(STATUS "  03-hpc-advanced:      ${BUILD_HPC_ADVANCED}")
message(STATUS "  04-inference-engine:  ${BUILD_INFERENCE_ENGINE}")
message(STATUS "")
message(STATUS "Options:")
message(STATUS "  BUILD_EXAMPLES:       ${BUILD_EXAMPLES}")
message(STATUS "  BUILD_TESTS:          ${BUILD_TESTS}")
message(STATUS "  BUILD_BENCHMARKS:     ${BUILD_BENCHMARKS}")
message(STATUS "  BUILD_PYTHON_BINDINGS:${BUILD_PYTHON_BINDINGS}")
message(STATUS "==========================================")
message(STATUS "")
