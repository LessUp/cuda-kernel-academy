# 04-inference-engine

轻量级 CUDA 推理引擎示例，演示如何把 GEMM、Tensor、内存管理和简单权重文件格式组合成一个可运行的推理流程。

## 当前模块包含的内容

- `InferenceEngine`：多层线性网络的加载、执行与计时
- `Tensor`：GPU Tensor 封装与基础算子
- `MemoryPool`：缓存式 GPU 内存池
- `StreamManager`：简单的多流管理器
- `Config` / `Logger`：配置与日志工具
- `mini_inference_benchmark` / `detailed_benchmark` / `mnist_demo`
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

从仓库根目录构建时，`04-inference-engine` 现在要求父项目先提供
`TensorCraft::tensorcraft` 目标；仓库根 `CMakeLists.txt` 已通过
`BUILD_TENSORCRAFT=ON` 保证这一点。该模块不再维护独立配置回退，也不再保留
“没有 TensorCraft 就退回本地 GEMM”的双路径构建逻辑。

### 从仓库根目录构建（推荐）

```bash
cmake --preset default
cmake --build --preset default
ctest --preset default
```

## 可用 CMake 选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `BUILD_TESTS` | ON | 构建 GoogleTest 测试 |

## 生成产物

- `mini_inference`：静态库
- `mini_inference_benchmark`：GEMM benchmark
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

## 创建测试权重文件

```cpp
#include "inference_engine.h"

using namespace mini_inference;

int main() {
    std::vector<std::pair<int, int>> layer_dims = {
        {784, 256},
        {256, 128},
        {128, 10}
    };
    create_random_weights("model.bin", layer_dims, true);
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

## 配置与日志

```cpp
#include "config.h"
#include "logger.h"

using namespace mini_inference;

int main() {
    Config::instance().load_from_file("config/default.ini");
    int device = Config::instance().get_int("CUDA_DEVICE", 0);

    Logger::instance().set_level(LogLevel::DEBUG);
    Logger::instance().set_file("app.log");
    LOG_INFO("Using device %d", device);
    return 0;
}
```

## 运行方式

```bash
cd build/default/bin
./mini_inference_benchmark
./detailed_benchmark
./mnist_demo
```

```bash
cd build/default
ctest --output-on-failure
```

如果你想直接运行测试二进制：

```bash
./build/default/bin/mini_inference_tests
```
