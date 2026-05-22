---
outline: [2, 3]
---

# Quick Start

Quick start guide for the Inference Engine module.

## Requirements

- Linux
- CUDA 12.x+
- CMake 3.20+
- NVIDIA GPU with CUDA support

## Build

```bash
cd 04-inference-engine
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Run

```bash
cd build
./benchmark
./detailed_benchmark
./mnist_demo
./mini_inference_tests
```

## Minimal Example

```cpp
#include "inference_engine.h"
#include "common.h"

using namespace mini_inference;

int main() {
    InferenceEngine engine;
    engine.init(0);
    engine.load_weights("model.bin");

    DeviceMemory d_input(batch_size * engine.input_dim() * sizeof(float));
    DeviceMemory d_output(batch_size * engine.output_dim() * sizeof(float));

    engine.forward(d_input.get(), d_output.get(), batch_size);
    engine.cleanup();
    return 0;
}
```

## References
