# 2026-01-08 构建系统稳定性修复

## 目的

提升仓库根目录“一键 CMake 构建”（superbuild）以及各子模块被 `add_subdirectory()` 引入时的稳定性与可复用性，避免路径跑偏、CUDA 架构不兼容、以及目标名冲突等问题。

## 变更概览

- `03-hpc-advanced/CMakeLists.txt`
  - 将 `cmake_minimum_required` 从 3.24 调整为 3.20（与仓库根与 CI 环境对齐）。
  - 修复 include 路径：将 `CMAKE_SOURCE_DIR` 改为 `CMAKE_CURRENT_SOURCE_DIR`，避免作为子目录构建时指向仓库根导致找不到源码。
  - 移除对仓库内不存在的 `FindCUDA/select_compute_arch` 的依赖，改为在未指定 `CMAKE_CUDA_ARCHITECTURES` 时使用默认值（避免配置阶段直接失败）。

- `03-hpc-advanced/python/CMakeLists.txt`
  - 修复 Python 绑定目标 include 路径：将 `${CMAKE_SOURCE_DIR}/src` 改为 `${PROJECT_SOURCE_DIR}/src`，避免顶层 superbuild 场景下路径跑偏。

- `04-inference-engine/CMakeLists.txt`
  - `CMAKE_CUDA_ARCHITECTURES`：仅在未指定时设置默认值，避免覆盖上层配置；默认值收敛为 CUDA 11 兼容集合。
  - 修复路径：避免误用 `CMAKE_SOURCE_DIR`，在独立构建时使用 `${CMAKE_CURRENT_LIST_DIR}/../02-tensorcraft-core` 作为依赖路径。
  - 依赖复用：若上层已提供 `tensorcraft` 目标则直接复用，否则仅在独立构建时 `add_subdirectory()` 引入。
  - 避免测试目标冲突：将测试可执行文件从通用名 `tests` 改为 `mini_inference_tests`。

- `02-tensorcraft-core/tests/CMakeLists.txt`
  - 修复 include 路径：将 `${CMAKE_SOURCE_DIR}/include` 改为 `${PROJECT_SOURCE_DIR}/include`，避免顶层 superbuild 场景下头文件路径跑偏。

- `02-tensorcraft-core/CMakeLists.txt`
  - `CMAKE_CUDA_ARCHITECTURES`：仅在未指定时设置默认值，并将默认架构收敛到 CUDA 11 兼容集合，避免在 CUDA 11 环境下因包含 `sm_89/sm_90` 导致配置/编译失败。

- `common/CMakeLists.txt`
  - 同步 CUDA 架构默认策略：仅在未指定时设置默认值，默认集合与仓库根一致。

- `CMakeLists.txt`（仓库根）
  - 默认 CUDA 架构收敛为 CUDA 11 兼容集合。
  - 将顶层 `BUILD_TESTS/BUILD_BENCHMARKS/BUILD_PYTHON_BINDINGS` 显式传递给 `02-tensorcraft-core` 的 `TC_BUILD_*` 选项，避免子模块默认开启 Python/测试导致意外拉取依赖。
  - 当 CUDA 编译器版本 < 12.0 时跳过 `03-hpc-advanced`（该模块要求 CUDA >= 12.0）。

- `CMakePresets.json`
  - 为兼容 `cmakeMinimumRequired` (3.20)，将若干使用 `CMAKE_CUDA_ARCHITECTURES: native` 的预设改为显式架构集合（避免在较旧 CMake 上配置失败）。

## 影响

- 根目录 CMake 构建更适合作为“聚合工程”使用。
- 各模块可以更可靠地被外部项目通过 `add_subdirectory()` 集成。
- 在 CUDA 11.x 环境下默认配置更不容易踩到不支持的 `sm_89/sm_90`。

## 验证说明

- 本地尝试执行 `cmake --preset minimal` 时，环境中未找到 `nvcc`（缺少 CUDA Toolkit），因此未能完成配置/编译验证。
