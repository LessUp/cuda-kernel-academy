# CUDA Kernel Academy - Agent Guide

High-signal guidance for working in this multi-module CUDA educational repository.

## Architecture

Four modules with **different build systems**:

| Module | Build System | C++ Standard | Notes |
|--------|-------------|--------------|-------|
| `01-sgemm-tutorial/` | **Standalone Makefile** | C++17 | Does NOT use CMake |
| `02-tensorcraft-core/` | CMake (header-only lib) | C++17/20/23 | Required by 04 |
| `03-hpc-advanced/` | CMake | **C++20 required** | Needs CUDA 12+, GCC 11+ or Clang 14+ |
| `04-inference-engine/` | CMake | C++17 | Depends on 02-tensorcraft-core |
| `common/` | CMake | C++17 | Shared utilities, header-only |

**Critical**: Module 01 uses its own Makefile. All others use CMake presets.

## Build Commands

### CMake Modules (02, 03, 04, common, examples)

```bash
# Standard workflow (recommended)
cmake --preset default
cmake --build --preset default
ctest --preset default

# Key presets: default, debug, release, native, minimal, full
# Architecture-specific: volta (sm_70), ampere (sm_80/86), ada (sm_89), hopper (sm_90)
cmake --list-presets  # See all options
```

### Module 01 (Standalone Makefile)

```bash
cd 01-sgemm-tutorial
make GPU_ARCH=sm_86        # Build benchmark
make test                   # Build and run tests
make clean
```

## Critical Constraints

- **03-hpc-advanced** auto-skips if CUDA < 12.0 or compiler lacks C++20 concepts support
- **04-inference-engine** fails CMake configure if `BUILD_TENSORCRAFT=OFF` (circular dependency)
- Use `native` preset to auto-detect host GPU architecture
- CI only runs CPU-safe checks (formatting, docs); **CUDA tests must run locally**

## Code Style

```bash
# Run before committing
pre-commit run --all-files
```

- **Format**: Google style via `.clang-format` (4 spaces, 100 cols, left pointer alignment)
- **Headers**: `#pragma once` (no include guards)
- **Namespaces**: `tensorcraft::`, `hpc::`, `mini_inference::`, `cuda_academy::`
- **Naming**: PascalCase classes, lower_case functions, trailing_underscore members_, kPascalCase global constants

## CUDA Error Handling

Always use the provided macros from `common/`:

```cpp
#include "cuda_academy/cuda_academy.hpp"

CUDA_CHECK(cudaMalloc(&ptr, size));
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

## Common CUDA Flags

Set in root `CMakeLists.txt`:
- `--expt-relaxed-constexpr` - constexpr in device code
- `--extended-lambda` - lambda capture in device code
- `--use_fast_math` - Release builds only

## Testing

```bash
# All tests
ctest --preset default

# Verbose
ctest --preset default -V

# Rerun failures
ctest --preset default --rerun-failed

# Single module (after build)
cd 02-tensorcraft-core && ctest --test-dir build --output-on-failure
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA not found | `export PATH=/usr/local/cuda/bin:$PATH; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH` |
| Wrong GPU arch | `nvidia-smi --query-gpu=compute_cap --format=csv` then `cmake --preset native` |
| GCC too old for 03 | Install GCC 11+: `cmake -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11` |

## Debug Tools

```bash
compute-sanitizer ./program    # Memory errors
nsys profile --stats=true ./prog  # System profile
ncu ./program                   # Kernel profile
cuda-gdb ./program              # GPU debugging
```

## Key File References

- Root build options: `CMakeLists.txt` lines 24-30
- Preset definitions: `CMakePresets.json`
- Format rules: `.clang-format`
- Pre-commit hooks: `.pre-commit-config.yaml`
- Common utilities: `common/include/cuda_academy/`
