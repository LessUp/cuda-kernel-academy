# Elementwise tests
hpc_add_test(test_relu test_relu.cpp)
target_link_libraries(test_relu PRIVATE hpc_elementwise)

hpc_add_test(test_sigmoid test_sigmoid.cpp)
target_link_libraries(test_sigmoid PRIVATE hpc_elementwise)

hpc_add_test(test_vector_add test_vector_add.cpp)
target_link_libraries(test_vector_add PRIVATE hpc_elementwise)

hpc_add_test(test_transpose test_transpose.cpp)
target_link_libraries(test_transpose PRIVATE hpc_elementwise)
