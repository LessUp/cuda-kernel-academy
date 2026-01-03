# CUDA 13 features tests
hpc_add_test(test_tma test_tma.cpp)
target_link_libraries(test_tma PRIVATE hpc_cuda13)

hpc_add_test(test_cluster test_cluster.cpp)
target_link_libraries(test_cluster PRIVATE hpc_cuda13)

hpc_add_test(test_fp8 test_fp8.cpp)
target_link_libraries(test_fp8 PRIVATE hpc_cuda13)
