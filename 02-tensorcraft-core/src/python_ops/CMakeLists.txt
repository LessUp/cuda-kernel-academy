# TensorCraft-HPC Python Bindings
pybind11_add_module(tensorcraft_ops
    bindings.cpp
    tensor_bindings.cpp
    kernel_bindings.cpp
)

target_link_libraries(tensorcraft_ops PRIVATE
    tensorcraft
    CUDA::cudart
)

set_target_properties(tensorcraft_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Install Python module
install(TARGETS tensorcraft_ops
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python
)
