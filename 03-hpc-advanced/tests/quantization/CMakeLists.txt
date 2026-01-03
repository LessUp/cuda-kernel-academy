# Quantization tests
hpc_add_test(test_quantize test_quantize.cpp)
target_link_libraries(test_quantize PRIVATE hpc_quantization)
