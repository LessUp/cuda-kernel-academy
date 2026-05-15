---
outline: [2, 3]
---

# 04-Inference Engine

轻量级 CUDA 推理引擎示例，演示如何把 GEMM、Tensor、内存管理和简单权重文件格式组合成一个可运行的推理流程。

## 当前模块包含的内容

- `InferenceEngine`：多层线性网络的加载、执行与计时
- `Tensor`：GPU Tensor 封装与基础算子
- `MemoryPool`：缓存式 GPU 内存池
- `StreamManager`：简单的多流管理器
- `Config` / `Logger`：配置与日志工具
- `benchmark` / `detailed_benchmark` / `mnist_demo`
- `mini_inference_tests`：统一测试目标

## 目录结构

```text
04-inference-engine/
├── include/
├── src/
├── benchmarks/
├── tests/
├── config/
├── docs/
├── scripts/
├── CMakeLists.txt
└── README.md
```

## 构建

从仓库根目录添加 `04-inference-engine` 时，模块现在要求父项目先提供现成的
`TensorCraft::tensorcraft` 目标；根 `CMakeLists.txt` 已在添加该模块前强制
`BUILD_TENSORCRAFT=ON` 来满足这一条件。

如果你是直接在 `04-inference-engine/` 目录下独立配置，模块仍会尝试从相邻的
`../02-tensorcraft-core` 自动补入 TensorCraft；只有这种独立构建场景才会使用该
仓库布局回退。若该目录不存在，则继续回退到本地 GEMM 实现。

### 从仓库根目录构建（推荐）

```bash
cmake --preset default
cmake --build --preset default
ctest --preset default
```

### 独立构建 04 模块

```bash
cd 04-inference-engine
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

## 可用 CMake 选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `BUILD_TESTS` | ON | 构建 GoogleTest 测试 |
| `USE_TENSORCRAFT` | ON | 优先链接父项目提供的 `TensorCraft::tensorcraft`；仅独立构建时才会尝试从 `../02-tensorcraft-core` 自动补入 |

## 生成产物

- `mini_inference`：静态库
- `benchmark`：GEMM benchmark
- `detailed_benchmark`：更细粒度 benchmark
- `mnist_demo`：随机权重的演示程序
- `mini_inference_tests`：统一测试二进制

## 基本示例：加载权重并前向执行

```cpp
#include "inference_engine.h"
#include "common.h"

using namespace mini_inference;

int main() {
    InferenceEngine engine;
    engine.init(0);

    if (!engine.load_weights("model.bin")) {
        return 1;
    }

    const int batch_size = 32;
    DeviceMemory d_input(batch_size * engine.input_dim() * sizeof(float));
    DeviceMemory d_output(batch_size * engine.output_dim() * sizeof(float));

    engine.forward(d_input.get(), d_output.get(), batch_size);
    engine.cleanup();
    return 0;
}
```

## Tensor API 示例

```cpp
#include "tensor.h"

using namespace mini_inference;

int main() {
    Tensor a({32, 784});
    Tensor w({784, 256});
    Tensor b({256});

    Tensor c = matmul(a, w);
    Tensor d = add_bias(c, b);
    Tensor e = relu(d);
    Tensor f = softmax(e);

    auto host = f.to_host();
    return 0;
}
```

## MemoryPool 与 StreamManager

### MemoryPool

```cpp
#include "memory_pool.h"

using namespace mini_inference;

int main() {
    auto& pool = MemoryPool::instance();
    void* ptr = pool.allocate(1 << 20);
    pool.deallocate(ptr);
    pool.print_stats();
    pool.clear_cache();
    return 0;
}
```

### StreamManager

```cpp
#include "stream_manager.h"

using namespace mini_inference;

int main() {
    auto& streams = StreamManager::instance();
    streams.init(4);
    cudaStream_t s0 = streams.get_stream();
    cudaStream_t s1 = streams.get_stream(1);
    streams.sync_all();
    streams.cleanup();
    return 0;
}
```

## 运行方式

```bash
cd build
./benchmark
./detailed_benchmark
./mnist_demo
ctest --output-on-failure
```

如果你想直接运行测试二进制：

```bash
./mini_inference_tests
```

## References

[^1]: NVIDIA. "CUDA C++ Programming Guide," v12.6. 2024. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
[^2]: NVIDIA cuBLAS. https://docs.nvidia.com/cuda/cublas/
[^3]: NVIDIA CUTLASS. https://github.com/NVIDIA/cutlass
