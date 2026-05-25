---
outline: [2, 3]
---

# Inference Engine 快速开始

本指南基于当前仓库中真实存在的 API、目标和构建方式。

## 环境要求

- Linux
- CUDA 12.x+（建议）
- CMake 3.20+
- 支持 CUDA 的 NVIDIA GPU

## 构建

```bash
cmake --preset default
cmake --build --preset default
ctest --preset default
```

## 运行生成产物

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

## 最小示例：InferenceEngine

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

    int batch_size = 32;
    DeviceMemory d_input(batch_size * engine.input_dim() * sizeof(float));
    DeviceMemory d_output(batch_size * engine.output_dim() * sizeof(float));

    engine.forward(d_input.get(), d_output.get(), batch_size);
    engine.cleanup();
    return 0;
}
```

## 生成随机权重

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

## Tensor 操作示例

```cpp
#include "tensor.h"

using namespace mini_inference;

int main() {
    Tensor a({32, 784});
    Tensor w({784, 256});
    Tensor b({256});

    Tensor out = relu(add_bias(matmul(a, w), b));
    auto host = out.to_host();
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
    Logger::instance().set_level(LogLevel::INFO);
    LOG_INFO("CUDA_DEVICE=%d", Config::instance().get_int("CUDA_DEVICE", 0));
    return 0;
}
```

## 多流管理

```cpp
#include "stream_manager.h"

using namespace mini_inference;

int main() {
    auto& manager = StreamManager::instance();
    manager.init(4);
    cudaStream_t stream = manager.get_stream();
    (void)stream;
    manager.sync_all();
    manager.cleanup();
    return 0;
}
```

## 说明

- 本模块的测试目标是统一的 `mini_inference_tests`，不是多个独立 `test_*` 可执行文件。
- 本模块通过仓库根构建，默认直接复用 `02-tensorcraft-core` 提供的 `TensorCraft::tensorcraft`。
- 如需完整 CUDA 验证，请在本地 GPU 环境运行。

## References
