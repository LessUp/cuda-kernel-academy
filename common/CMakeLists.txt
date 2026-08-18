cmake_minimum_required(VERSION 3.20)
project(cuda-foundations-common LANGUAGES CXX CUDA)

# ============================================================================
# C++ Standard Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# CUDA Configuration
# ============================================================================
find_package(CUDAToolkit REQUIRED)

# CUDA architectures (Volta to Hopper)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES
      "70;75;80;86"
      CACHE STRING "CUDA architectures")
endif()

# ============================================================================
# Header-Only Library
# ============================================================================
add_library(cuda_foundations_common INTERFACE)
add_library(CUDAFoundations::common ALIAS cuda_foundations_common)

target_include_directories(
  cuda_foundations_common
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)

target_link_libraries(cuda_foundations_common INTERFACE CUDA::cudart CUDA::cublas)

# ============================================================================
# Installation
# ============================================================================
include(GNUInstallDirs)

install(
  TARGETS cuda_foundations_common
  EXPORT CUDAFoundationsTargets
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "CUDA Academy Common Library Configuration:")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Archs:     ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "")
