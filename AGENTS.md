# CUDA Kernel Academy - Agent Guide

This file contains essential information for AI coding agents working on the **CUDA Kernel Academy** project.

---

## Project Overview

CUDA Kernel Academy is a structured educational repository for learning high-performance CUDA kernel development. It covers matrix operations, AI kernels, and inference engine implementation with progressive complexity.

### Learning Path

```
01-sgemm-tutorial (fundamentals)
       ↓
02-tensorcraft-core (reusable library)
       ↓
03-hpc-advanced (expert techniques)  OR  04-inference-engine (application)
```

### Module Summary

| # | Module | Description | Build System | C++ Standard |
|---|--------|-------------|--------------|--------------|
| 01 | `01-sgemm-tutorial/` | SGEMM optimization tutorial | Standalone Makefile | C++17 |
| 02 | `02-tensorcraft-core/` | Header-only kernel library | CMake | C++17/20/23 |
| 03 | `03-hpc-advanced/` | Advanced CUDA/HPC techniques | CMake | C++20 |
| 04 | `04-inference-engine/` | Lightweight DL inference engine | CMake | C++17 |
| - | `common/` | Shared utilities (header-only) | CMake | C++17 |
| - | `examples/` | Integration examples | CMake | C++17 |

---

## Technology Stack

### Core Technologies

- **CUDA Toolkit**: 11.0+ (modules 01, 02, 04), 12.0+ (module 03)
- **CMake**: 3.20+ (minimum), 3.24+ (recommended)
- **C++ Standards**: C++17 (default), C++20 (module 03 required)
- **Compilers**: GCC 9+ or Clang 10+ (GCC 11+ recommended for C++20)

### GPU Architecture Support

| Architecture | sm | Example GPUs |
|--------------|-----|--------------|
| Volta | sm_70 | V100 |
| Turing | sm_75 | RTX 2080, T4 |
| Ampere | sm_80, sm_86 | A100, RTX 3090 |
| Ada | sm_89 | RTX 4090, L40 |
| Hopper | sm_90 | H100 (TMA/Cluster features) |

### Dependencies (Auto-fetched via CMake)

| Module | Dependencies |
|--------|-------------|
| 02-tensorcraft-core | GoogleTest v1.14.0, google/benchmark v1.8.3, pybind11 v2.11.1 |
| 03-hpc-advanced | GoogleTest v1.14.0, fmt 10.2.1, rapidcheck, CUTLASS v3.5.0, nanobind v2.0.0 |
| 04-inference-engine | GoogleTest v1.14.0 (links to tensorcraft-core) |
| 01-sgemm-tutorial | System GoogleTest (`apt install libgtest-dev`) |

---

## Build System

### Quick Build (Recommended)

```bash
# Configure, build, and test with presets
cmake --preset default
cmake --build --preset default
ctest --preset default
```

### Available Presets

| Preset | CUDA Archs | Build Type | Notes |
|--------|------------|------------|-------|
| `default` | 70;80;86 | Release | Standard build |
| `debug` | 70;80;86 | Debug | Debug symbols, no fast-math |
| `release` | 70;75;80;86 | Release | All common architectures |
| `volta` | 70 | Release | V100 only |
| `ampere` | 80;86 | Release | A100, RTX 30 series |
| `ada` | 89 | Release | RTX 40 series |
| `hopper` | 90 | Release | H100 (requires CUDA 12+) |
| `native` | native | Release | Host GPU only |
| `minimal` | 70;80;86 | Release | Only 02-tensorcraft-core |
| `full` | 70;75;80;86;89 | Release | Everything including Python bindings |

### Root CMake Build Options

```cmake
option(BUILD_TENSORCRAFT "Build 02-tensorcraft-core module" ON)
option(BUILD_HPC_ADVANCED "Build 03-hpc-advanced module" ON)
option(BUILD_INFERENCE_ENGINE "Build 04-inference-engine module" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
```

### Module-Specific Builds

**01-sgemm-tutorial** (Standalone Makefile):

```bash
cd 01-sgemm-tutorial
make GPU_ARCH=sm_86        # Build benchmark
make test                   # Build and run tests
make clean                  # Clean build
```

**Individual module with CMake**:

```bash
cd 02-tensorcraft-core
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

---

## Testing Strategy

### Test Frameworks

| Module | Primary Framework | Additional Tools |
|--------|-------------------|------------------|
| 01-sgemm-tutorial | GoogleTest | Parameterized tests |
| 02-tensorcraft-core | GoogleTest + Google Benchmark | CTest integration |
| 03-hpc-advanced | GoogleTest + RapidCheck | Property-based testing |
| 04-inference-engine | GoogleTest | Component tests |

### Running Tests

```bash
# All modules (after preset build)
ctest --preset default

# With verbose output
ctest --preset default -V

# Run only failed tests
ctest --preset default --rerun-failed

# Specific module
cd 02-tensorcraft-core && ctest --test-dir build
```

### Test Files by Module

- **01-sgemm-tutorial**: `tests/test_sgemm.cu`
- **02-tensorcraft-core**: `test_main.cpp`, `test_elementwise.cpp`, `test_softmax.cpp`, `test_normalization.cpp`, `test_gemm.cpp`
- **03-hpc-advanced**: Organized by category in `tests/elementwise/`, `tests/reduction/`, `tests/gemm/`, `tests/attention/`, etc.
- **04-inference-engine**: `test_advanced.cpp`, `test_config.cpp`, `test_fusion.cpp`, `test_gemm.cpp`, `test_inference.cpp`, `test_logger.cpp`, `test_memory_pool.cpp`, `test_quantization.cpp`, `test_stream_manager.cpp`, `test_tensor.cpp`

---

## Code Style Guidelines

### Formatting Rules (`.clang-format`)

| Setting | Value |
|---------|-------|
| Base style | Google |
| Indent width | 4 spaces |
| Tab width | 4 |
| Column limit | 100 characters |
| Braces | Attach (K&R style) |
| Pointer alignment | Left (`int* ptr`) |
| Short functions | Inline only |
| Include sorting | Yes (CUDA → System → Project) |

### File Naming Conventions

| File Type | Extension |
|-----------|-----------|
| C++ headers | `.hpp` |
| CUDA headers | `.cuh` |
| C++ sources | `.cpp` |
| CUDA sources | `.cu` |
| Test files | `test_*.cpp` |

### Naming Conventions

| Entity | Style | Example |
|--------|-------|---------|
| Classes/Structs | PascalCase | `InferenceEngine`, `Tensor` |
| Functions | lower_case | `launch_gemm_kernel` |
| Variables | lower_case | `block_size`, `num_threads` |
| Member variables | trailing underscore | `value_`, `size_` |
| Constants (global) | kPascalCase | `kBlockSize`, `kMaxThreads` |
| Constants (local) | UPPER_CASE | `PI`, `TOLERANCE` |
| Namespaces | lower_case | `tensorcraft`, `hpc` |
| Macros | UPPER_CASE | `CUDA_CHECK` |

### Namespace Conventions

| Module | Namespace |
|--------|-----------|
| 01-sgemm-tutorial | `sgemm::` |
| 02-tensorcraft-core | `tensorcraft::` |
| 03-hpc-advanced | `hpc::` |
| 04-inference-engine | `mini_inference::` |
| Common utilities | `cuda_academy::` |

### Include Guards

Use `#pragma once` instead of include guards.

### Documentation

Doxygen-style comments required for public APIs:

```cpp
/**
 * @brief Brief description
 * @param param Parameter description
 * @return Return value description
 */
```

### EditorConfig

- Default: 4 spaces, UTF-8, LF line endings, trim trailing whitespace
- YAML/JSON/Markdown: 2 spaces
- Makefile: tabs required
- Python: 4 spaces

---

## Pre-commit Hooks

All code must pass pre-commit checks before committing.

### Installation

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Enabled Hooks

1. **General** (`pre-commit-hooks`):
   - trailing-whitespace, end-of-file-fixer
   - check-yaml, check-json
   - check-added-large-files (max 1000KB)
   - check-merge-conflict, check-case-conflict
   - mixed-line-ending (fix to LF)

2. **C/C++/CUDA** (`mirrors-clang-format` v17.0.6)
3. **CMake** (`cmake-format-precommit` v0.6.13)
4. **Markdown** (`markdownlint-cli` v0.38.0)
5. **YAML** (`yamllint` v1.33.0)
6. **Shell** (`shellcheck-py` v0.9.0.6)
7. **Python** (`black` v24.1.1, `isort` v5.13.2)

### Clang-Tidy Checks

**Enabled**: `bugprone-*`, `clang-analyzer-*`, `cppcoreguidelines-*`, `modernize-*`, `performance-*`, `readability-*`

**Disabled** (HPC context):

- `modernize-use-trailing-return-type`
- `readability-magic-numbers`
- `cppcoreguidelines-avoid-magic-numbers`
- `cppcoreguidelines-pro-bounds-pointer-arithmetic`
- `cppcoreguidelines-avoid-c-arrays`, `modernize-avoid-c-arrays`
- `readability-identifier-length`
- `bugprone-easily-swappable-parameters`

---

## CI/CD Workflows

### `.github/workflows/ci.yml`

Runs on push/PR to main/master:

| Job | Purpose |
|-----|---------|
| `pre-commit` | Run all pre-commit hooks |
| `docs-build` | Build documentation with HonKit |
| `markdown-links` | Check for broken links |
| `cmake-presets` | Validate CMake presets |

### `.github/workflows/pages.yml`

Deploys documentation to GitHub Pages when docs change.

**Note**: CI runs CPU-safe checks only. Full CUDA builds/tests must be run locally on a GPU machine.

---

## Project Directory Structure

```
cuda-kernel-academy/
├── 01-sgemm-tutorial/          # Standalone SGEMM tutorial
│   ├── src/                    # Kernel implementations
│   ├── tests/                  # GoogleTest tests
│   └── Makefile                # Standalone build
│
├── 02-tensorcraft-core/        # Header-only kernel library
│   ├── include/tensorcraft/    # Public API headers
│   │   ├── core/               # CUDA error checking, type traits
│   │   ├── kernels/            # GEMM, attention, normalization
│   │   └── memory/             # Tensor wrapper & memory pool
│   ├── benchmarks/             # Google Benchmark tests
│   ├── tests/                  # Unit tests
│   └── src/python_ops/         # Python bindings (pybind11)
│
├── 03-hpc-advanced/            # Advanced CUDA/HPC techniques
│   ├── src/
│   │   ├── 01_elementwise/     # Vector ops, ReLU, sigmoid
│   │   ├── 02_reduction/       # LayerNorm, RMSNorm, softmax
│   │   ├── 03_gemm/            # GEMM optimization
│   │   ├── 04_convolution/     # Winograd, implicit GEMM
│   │   ├── 05_attention/       # FlashAttention, RoPE, TopK
│   │   ├── 06_quantization/    # INT8, FP8 quantization
│   │   └── 07_cuda13_features/ # TMA, thread blocks, FP8
│   ├── python/                 # Python bindings
│   ├── tests/                  # Categorized tests
│   └── docker/                 # Dev container configs
│
├── 04-inference-engine/        # DL inference engine
│   ├── include/                # Public headers
│   ├── src/                    # Implementation
│   ├── benchmarks/             # Performance benchmarks
│   ├── tests/                  # Unified tests
│   ├── config/                 # Configuration files
│   └── scripts/                # Utility scripts
│
├── common/                     # Shared utilities (header-only)
│   └── include/cuda_academy/   # Error checking, timers
│
├── examples/                   # Integration examples
├── docs/                       # Project documentation
├── scripts/                    # Build/post-build scripts
├── .github/workflows/          # CI/CD configurations
└── CMakeLists.txt              # Root CMake configuration
```

---

## Development Conventions

### Commit Messages

Use Conventional Commits:

```
feat(scope): short description
fix(scope): short description
docs(scope): short description
ci(scope): short description
```

### CUDA Error Handling

Always use the provided error checking macros:

```cpp
#include "cuda_academy/cuda_academy.hpp"

CUDA_CHECK(cudaMalloc(&ptr, size));
CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

### CUDA Compilation Flags

Common flags used:

- `--expt-relaxed-constexpr` - Allow constexpr in device code
- `--extended-lambda` - Allow lambda capture in device code
- `--use_fast_math` - Enable fast math (Release builds)

### Memory Management

- Prefer `cuda_academy` utilities for RAII wrappers
- Use `cudaMallocAsync`/`cudaFreeAsync` with streams when possible
- Check memory availability with `cudaMemGetInfo`

---

## Troubleshooting Common Issues

### CUDA Not Found

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Unsupported GCC Version

```bash
# Use compatible GCC version
sudo apt install gcc-11 g++-11
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
         -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11
```

### Wrong GPU Architecture

```bash
# Find your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Build for specific architecture
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86  # RTX 3090
```

### Debug Tools

| Tool | Command | Purpose |
|------|---------|---------|
| compute-sanitizer | `compute-sanitizer ./program` | Memory error detection |
| cuda-gdb | `cuda-gdb ./program` | GPU debugging |
| nsys | `nsys profile --stats=true ./program` | System profiling |
| ncu | `ncu ./program` | Kernel profiling |

---

## Security Considerations

1. **CUDA Memory Safety**: Always validate array bounds in kernels
2. **Kernel Launch Bounds**: Use `__launch_bounds__` to prevent register overflow
3. **Input Validation**: Sanitize inputs to kernels to prevent out-of-bounds access
4. **No Secrets**: Never commit API keys, passwords, or credentials
5. **Large Files**: Pre-commit blocks files > 1000KB

---

## Documentation

### Project Documentation (`docs/`)

| Document | Purpose |
|----------|---------|
| `INSTALLATION.md` | Build instructions, system requirements |
| `CODING_STYLE.md` | Detailed coding standards |
| `TROUBLESHOOTING.md` | Common issues and solutions |
| `FAQ.md` | Frequently asked questions |
| `DEPLOYMENT.md` | GitHub Pages deployment |
| `integration_examples.md` | Integration examples |

### Website Documentation (HonKit)

- Built with HonKit
- Deployed to GitHub Pages
- Source: `SUMMARY.md`, `book.json`, `assets/`
- Build: `npm ci && npm run docs:build`

---

## License

MIT License - See `LICENSE` file for details.

---

## Quick Reference Commands

```bash
# Full build
cmake --preset default && cmake --build --preset default

# Run tests
ctest --preset default

# Format code
pre-commit run --all-files

# Build docs
npm ci && npm run docs:build

# List available presets
cmake --list-presets

# Build single module (e.g., SGEMM tutorial)
cd 01-sgemm-tutorial && make GPU_ARCH=sm_86 && make test

# Debug build with symbols
cmake --preset debug && cmake --build --preset debug
```
