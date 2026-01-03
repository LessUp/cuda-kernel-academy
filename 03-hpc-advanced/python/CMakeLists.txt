# Python bindings using nanobind
if(BUILD_PYTHON_BINDINGS)
    nanobind_add_module(hpc_kernels
        bindings/bindings.cpp
    )
    
    target_link_libraries(hpc_kernels PRIVATE
        hpc_elementwise
        hpc_reduction
        hpc_gemm
        hpc_attention
    )
    
    target_include_directories(hpc_kernels PRIVATE ${CMAKE_SOURCE_DIR}/src)
    
    install(TARGETS hpc_kernels LIBRARY DESTINATION .)
endif()
