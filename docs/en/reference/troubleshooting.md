---
outline: [2, 3]
---

# Troubleshooting

Common issues and solutions for CUDA Kernel Academy.

## Compilation Errors

### nvcc: command not found

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### GCC Version Mismatch

| CUDA Version | Supported GCC |
|--------------|---------------|
| CUDA 11.x | GCC ≤ 10 |
| CUDA 12.0-12.2 | GCC ≤ 12 |
| CUDA 12.3+ | GCC ≤ 13 |

```bash
cmake .. -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11
```

### GPU Architecture Mismatch

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86
```

## Runtime Errors

### CUDA error: out of memory

```bash
nvidia-smi  # Check GPU memory usage
```

### CUDA error: illegal memory access

```bash
compute-sanitizer --tool memcheck ./your_program
```

## Module-Specific Issues

### 03-hpc-advanced build fails

Requires CUDA 12.0+ and C++20 support.

```bash
nvcc --version
```

### 04-inference-engine link errors

Depends on tensorcraft-core.

```bash
cmake .. -DBUILD_TENSORCRAFT=ON
```

## Debugging Tools

| Tool | Purpose | Command |
|------|---------|---------|
| compute-sanitizer | Memory check | `compute-sanitizer ./program` |
| nsys | System profiling | `nsys profile ./program` |
| ncu | Kernel profiling | `ncu ./program` |

## References
