# TensorCraft-HPC Benchmarks

# GEMM Benchmark
add_executable(gemm_benchmark gemm_benchmark.cpp)
target_link_libraries(gemm_benchmark PRIVATE
    tensorcraft
    benchmark::benchmark
    benchmark::benchmark_main
    CUDA::cudart
)
set_source_files_properties(gemm_benchmark.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(gemm_benchmark PROPERTIES LINKER_LANGUAGE CUDA)

# Attention Benchmark
add_executable(attention_benchmark attention_benchmark.cpp)
target_link_libraries(attention_benchmark PRIVATE
    tensorcraft
    benchmark::benchmark
    benchmark::benchmark_main
    CUDA::cudart
)
set_source_files_properties(attention_benchmark.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(attention_benchmark PROPERTIES LINKER_LANGUAGE CUDA)

# Conv Benchmark
add_executable(conv_benchmark conv_benchmark.cpp)
target_link_libraries(conv_benchmark PRIVATE
    tensorcraft
    benchmark::benchmark
    benchmark::benchmark_main
    CUDA::cudart
)
set_source_files_properties(conv_benchmark.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(conv_benchmark PROPERTIES LINKER_LANGUAGE CUDA)
