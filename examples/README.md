# CUDA Kernel Academy Examples

This directory contains small standalone examples built from the root CMake project.

## Available examples

| Example | Description |
|---------|-------------|
| [01_basic_gemm](01_basic_gemm/) | Basic GEMM example |
| [02_tensor_operations](02_tensor_operations/) | Tensor operation example |

## Build all examples

From the repository root:

```bash
cmake --preset default
cmake --build --preset default
```

If you only want examples, use a manual configure:

```bash
cmake -S . -B build/examples -DBUILD_EXAMPLES=ON
cmake --build build/examples -j$(nproc)
```

## Build a single example

```bash
cd examples/01_basic_gemm
cmake -S . -B build
cmake --build build
./build/basic_gemm
```

## Requirements

- CUDA Toolkit 12.x+
- CMake 3.20+
- NVIDIA GPU with SM 7.0+

## Related docs

- [Installation Guide](../docs/INSTALLATION.md)
- [Coding Style](../docs/CODING_STYLE.md)
- [Troubleshooting](../docs/TROUBLESHOOTING.md)
