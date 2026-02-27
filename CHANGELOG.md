# Changelog

All notable changes to CUDA Kernel Academy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

### Changed
- Nothing yet

### Fixed
- Nothing yet

## [1.0.0] - 2026-01-07

### Added

#### Project Structure
- Comprehensive 4-module learning path for CUDA kernel development
- Unified project organization with shared common utilities
- Complete documentation system with coding style guide

#### 01-sgemm-tutorial
- Naive SGEMM implementation (baseline)
- Shared memory tiled SGEMM
- Bank conflict-free SGEMM optimization
- Double buffering SGEMM
- Tensor Core SGEMM using WMMA API
- Benchmark suite comparing all implementations against cuBLAS
- Correctness verification utilities

#### 02-tensorcraft-core
- Header-only high-performance kernel library
- GEMM implementations (Naive, Tiled, Double Buffer, Tensor Core)
- Attention mechanisms (Flash Attention style, RoPE)
- Normalization layers (LayerNorm, RMSNorm, BatchNorm)
- Elementwise operations (ReLU, GeLU, SiLU, Sigmoid)
- Softmax implementation
- Conv2D operations
- Sparse matrix operations (SpMV, SpMM)
- Operator fusion utilities
- Python bindings via pybind11
- Comprehensive test suite with GoogleTest
- Performance benchmarks

#### 03-hpc-advanced
- 7-step GEMM optimization path
- Register tiling implementation
- Software pipelining techniques
- CUDA 13 features support:
  - TMA (Tensor Memory Accelerator)
  - Thread Block Clusters
  - FP8 GEMM
- LLM-specific operators (FlashAttention, RoPE, MoE TopK)
- Property-based testing with RapidCheck
- Python bindings via Nanobind
- Docker environment for CUDA 13.1

#### 04-inference-engine
- Complete inference engine framework
- GPU Memory Pool management
- CUDA Stream Manager for concurrent execution
- Auto-tuner for kernel configuration
- Profiler for performance analysis
- INT8/FP16 quantization support
- MNIST demo application
- Configuration system (INI-based)
- Logging system

#### Documentation
- Main README with learning path visualization
- Coding style guide (docs/CODING_STYLE.md)
- Integration examples
- Per-module documentation
- API references

#### Build System
- CMake-based build for all modules except 01-sgemm-tutorial
- Makefile for 01-sgemm-tutorial (beginner-friendly)
- Support for CUDA architectures sm_70 through sm_90
- CMake presets for common configurations

### Technical Details

- **C++ Standard**: C++17 minimum, C++20 for advanced modules
- **CUDA Standard**: CUDA 17
- **Supported Architectures**: Volta (sm_70) to Hopper (sm_90)
- **Testing Framework**: GoogleTest
- **Benchmarking**: Google Benchmark
- **Property Testing**: RapidCheck (03-hpc-advanced)

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2026-01-07 | Initial release with 4 learning modules |

[Unreleased]: https://github.com/LessUp/cuda-kernel-academy/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/LessUp/cuda-kernel-academy/releases/tag/v1.0.0
