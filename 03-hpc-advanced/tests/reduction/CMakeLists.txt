# Reduction tests
hpc_add_test(test_softmax test_softmax.cpp)
target_link_libraries(test_softmax PRIVATE hpc_reduction)

hpc_add_test(test_layernorm test_layernorm.cpp)
target_link_libraries(test_layernorm PRIVATE hpc_reduction)
