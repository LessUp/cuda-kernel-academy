cmake_minimum_required(VERSION 3.20)
project(cuda_kernel_academy_examples LANGUAGES CXX CUDA)

# C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 架构 (如果未在父项目中设置)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;80;86")
endif()

# 添加子目录
add_subdirectory(01_basic_gemm)
add_subdirectory(02_tensor_operations)

# 打印配置信息
message(STATUS "Building CUDA Kernel Academy Examples")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
