# Attention tests
hpc_add_test(test_flash_attention test_flash_attention.cpp)
target_link_libraries(test_flash_attention PRIVATE hpc_attention)

hpc_add_test(test_rope test_rope.cpp)
target_link_libraries(test_rope PRIVATE hpc_attention)

hpc_add_test(test_topk test_topk.cpp)
target_link_libraries(test_topk PRIVATE hpc_attention)
