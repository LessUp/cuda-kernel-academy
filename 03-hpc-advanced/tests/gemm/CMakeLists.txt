# GEMM tests
hpc_add_test(test_gemm test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE hpc_gemm)
