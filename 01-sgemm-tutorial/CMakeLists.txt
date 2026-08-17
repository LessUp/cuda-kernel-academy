# =============================================================================
# 01-sgemm-tutorial CMake integration
#
# The Makefile is still available for quick standalone use.  This target makes
# the tutorial part of the repository-wide `ctest` run so it can no longer rot
# outside CI.
# =============================================================================

find_package(CUDAToolkit REQUIRED)

add_executable(sgemm_tutorial_benchmark src/main.cu)
target_include_directories(sgemm_tutorial_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(sgemm_tutorial_benchmark PRIVATE --use_fast_math -lineinfo)
target_link_libraries(sgemm_tutorial_benchmark PRIVATE CUDA::cudart CUDA::cublas
                                                        CUDA::curand)

if(BUILD_TESTS)
  if(NOT TARGET GTest::gtest)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
      URL_HASH MD5=c8340a482851ef6a3fe618a082304cfc
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  add_executable(sgemm_tutorial_tests tests/test_sgemm.cu)
  set_source_files_properties(tests/test_sgemm.cu PROPERTIES LANGUAGE CUDA)
  target_include_directories(sgemm_tutorial_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_options(sgemm_tutorial_tests PRIVATE --use_fast_math -lineinfo)
  target_link_libraries(sgemm_tutorial_tests PRIVATE GTest::gtest CUDA::cudart
                                                    CUDA::cublas CUDA::curand)
  set_target_properties(sgemm_tutorial_tests PROPERTIES LINKER_LANGUAGE CUDA)

  include(GoogleTest)
  gtest_discover_tests(sgemm_tutorial_tests)
endif()
