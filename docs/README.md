# CUDA Kernel Academy Documentation

Welcome to the CUDA Kernel Academy documentation center.

## Documentation Index

### Getting Started

- [Main README](../README.md) - Project overview and quick start
- [Installation Guide](INSTALLATION.md) - Detailed installation and build instructions
- [FAQ](FAQ.md) - Frequently asked questions
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions

### Project Standards

- [Coding Style Guide](CODING_STYLE.md) - Unified code style and technical standards
- [Integration Examples](integration_examples.md) - How to integrate modules in your project

### Example Code

- [Examples Directory](../examples/README.md) - Standalone runnable example code

### Module Documentation

| Module | Documentation | Description |
|--------|---------------|-------------|
| 01-sgemm-tutorial | [README](../01-sgemm-tutorial/README.md) | SGEMM optimization tutorial |
| 02-tensorcraft-core | [docs/](../02-tensorcraft-core/docs/) | Core kernel library documentation |
| 03-hpc-advanced | [docs/](../03-hpc-advanced/docs/) | Advanced optimization documentation |
| 04-inference-engine | [docs/](../04-inference-engine/docs/) | Inference engine documentation |

### Technical Deep Dives

#### 02-tensorcraft-core

- [architecture.md](../02-tensorcraft-core/docs/architecture.md) - Architecture design
- [api_reference.md](../02-tensorcraft-core/docs/api_reference.md) - API reference
- [optimization_guide.md](../02-tensorcraft-core/docs/optimization_guide.md) - Optimization guide
- [modern_cpp_cuda.md](../02-tensorcraft-core/docs/modern_cpp_cuda.md) - Modern C++ CUDA

#### 03-hpc-advanced

- [01_gemm_optimization.md](../03-hpc-advanced/docs/01_gemm_optimization.md) - GEMM optimization details
- [02_memory_optimization.md](../03-hpc-advanced/docs/02_memory_optimization.md) - Memory optimization
- [03_reduction_optimization.md](../03-hpc-advanced/docs/03_reduction_optimization.md) - Reduction optimization
- [04_flash_attention.md](../03-hpc-advanced/docs/04_flash_attention.md) - Flash Attention
- [05_cuda13_features.md](../03-hpc-advanced/docs/05_cuda13_features.md) - CUDA 13 new features

#### 04-inference-engine

- [ARCHITECTURE.md](../04-inference-engine/docs/ARCHITECTURE.md) - Architecture design
- [API_REFERENCE.md](../04-inference-engine/docs/API_REFERENCE.md) - API reference
- [QUICK_START.md](../04-inference-engine/docs/QUICK_START.md) - Quick start guide
- [PERFORMANCE_TUNING.md](../04-inference-engine/docs/PERFORMANCE_TUNING.md) - Performance tuning

## Recommended Learning Path

### Beginners (1-2 weeks)

1. Read the main README to understand the project structure
2. Navigate to `01-sgemm-tutorial` and learn the 5 GEMM optimization techniques in order
3. Run benchmarks to understand performance differences

### Intermediate (2-4 weeks)

1. Study the API design of `02-tensorcraft-core`
2. Read `architecture.md` to understand design principles
3. Try integrating tensorcraft into your own project

### Advanced (4-8 weeks)

1. Dive into CUDA 13 features in `03-hpc-advanced`
2. Study Flash Attention implementation
3. Build a complete inference engine

## Contributing

Contributions to improve documentation are welcome! Please refer to [CONTRIBUTING.md](../CONTRIBUTING.md) in the root directory.
