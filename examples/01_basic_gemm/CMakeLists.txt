cmake_minimum_required(VERSION 3.20)
project(basic_gemm LANGUAGES CXX CUDA)

# C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 架构 (默认支持 Volta 到 Hopper)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;80;86")
endif()

# 创建可执行文件
add_executable(basic_gemm main.cu)

# 编译选项
target_compile_options(basic_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcompiler=-Wall
    >
)

# Release 优化
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(basic_gemm PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
        >
    )
endif()
