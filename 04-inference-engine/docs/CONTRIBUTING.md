# 贡献指南

感谢你对 `04-inference-engine` 的关注。

## 获取代码

```bash
git clone https://github.com/LessUp/cuda-kernel-academy.git
cd cuda-kernel-academy/04-inference-engine
```

## 构建与测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

也可以直接运行统一测试二进制：

```bash
./build/mini_inference_tests
```

## 运行程序

```bash
./build/benchmark
./build/detailed_benchmark
./build/mnist_demo
```

## 修改代码时请注意

- 以 `include/` 中的公开头文件为准维护 README 和示例
- 当前公开 CMake 选项只有：
  - `BUILD_TESTS`
  - `USE_TENSORCRAFT`
- 测试目标是统一的 `mini_inference_tests`
- 若修改了命令、目标名或 API，请同步更新 `README.md` 与 `docs/QUICK_START.md`

## 建议本地检查

```bash
ctest --test-dir build --output-on-failure
```

如修改了更上层文档或构建配置，请在仓库根目录补充运行：

```bash
pre-commit run --all-files
npm ci
npx honkit build . _site
```
