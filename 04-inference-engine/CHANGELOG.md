# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Stream Manager for multi-stream concurrent execution
- Batched GEMM operations with stream parallelism
- INT8 quantization support with per-channel quantization
- Quantization calibrator for dynamic quantization
- Comprehensive documentation suite
  - Quick Start Guide
  - Architecture documentation
  - GEMM Optimization detailed guide
  - Performance Tuning guide
  - API Reference
  - Contributing guide
- Configuration file examples (default, high_performance, debug)

## [1.0.0] - 2024-12-31

### Added
- Complete GEMM optimization implementation
  - Level 1: Naive MatMul
  - Level 2: Tiled GEMM with shared memory
  - Level 3: Memory coalescing optimization
  - Level 4: Double buffering for latency hiding
  - Level 5: Register blocking with vectorized loads
  - Level 6: Kernel fusion (MatMul + Bias + ReLU)
  - Level 7: Vectorized GEMM with float4 loads
- Half precision (FP16) GEMM with mixed precision accumulation
- Auto-tuner for automatic kernel selection
- Performance profiler with roofline analysis
- N-dimensional Tensor class with GPU storage
- GPU memory pool with caching
- Thread-safe logging system
- Configuration management system
- InferenceEngine for neural network inference
- MNIST demo application
- Comprehensive benchmark suite
- Test suite with Google Test

### Infrastructure
- CMake build system with CUDA support
- Optional test building (BUILD_TESTS flag)
- Support for CUDA architectures 75, 80, 86, 89, 90

## [0.1.0] - 2024-12-30

### Added
- Initial project structure
- Basic GEMM kernels (Naive, Tiled)
- Core data structures (MatrixDesc, GemmConfig)
- CUDA error handling utilities
- DeviceMemory RAII wrapper
- Basic benchmark framework

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2024-12-31 | Complete implementation with all optimizations |
| 0.1.0 | 2024-12-30 | Initial release with basic functionality |

## Migration Guide

### From 0.1.0 to 1.0.0

No breaking changes. New features are additive.

### API Changes

None.

### Deprecated Features

None.

## Known Issues

- Performance on small matrices (< 256) may be suboptimal
- FP16 GEMM requires GPU with Compute Capability 7.0+
- Memory pool may hold onto memory longer than necessary

## Roadmap

### Planned for 1.1.0
- [ ] Tensor Core support (WMMA API)
- [ ] Multi-GPU support
- [ ] ONNX model loading

### Planned for 1.2.0
- [ ] INT4 quantization
- [ ] Sparse matrix support
- [ ] Custom activation functions

### Long-term
- [ ] Transformer layer support
- [ ] Convolution operations
- [ ] Graph optimization
