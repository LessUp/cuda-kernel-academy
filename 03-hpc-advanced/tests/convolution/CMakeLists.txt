# Convolution tests
hpc_add_test(test_conv test_conv.cpp)
target_link_libraries(test_conv PRIVATE hpc_convolution)
