# CUDA Kernel Academy - Agent Guide

Keep agent behavior simple: read the repository, make narrowly scoped changes, and leave the tracked tree easier to understand than you found it.

## Project shape

| Module | Build system | Standard | Notes |
| --- | --- | --- | --- |
| `01-sgemm-tutorial/` | standalone `Makefile` | C++17 | not part of root CMake |
| `02-tensorcraft-core/` | root/module CMake | C++17+ | header-only, reused by 04 |
| `03-hpc-advanced/` | root/module CMake | C++20 | requires CUDA 12+ and modern host compiler |
| `04-inference-engine/` | root/module CMake | C++17 | depends on 02 |
| `common/` | root CMake | C++17 | shared CUDA helpers |

## Working posture

- favor cleanup, correctness, and documentation clarity over feature growth
- prefer deleting stale material over adding another framework or process layer
- keep GitHub Actions CPU-safe and repository-specific
- keep GitHub Pages focused on learning and reference content

## Build and verification

```bash
cmake --list-presets
cmake --preset default
cmake --build --preset default
ctest --preset default

pre-commit run --all-files
npm ci
npm run docs:build
```

For module 01:

```bash
cd 01-sgemm-tutorial
make GPU_ARCH=sm_86
make test
```

Notes:

- CUDA runtime validation belongs on a real GPU machine
- `03-hpc-advanced` may be unavailable without CUDA 12+ and a modern host compiler
- `04-inference-engine` depends on `02-tensorcraft-core`

## Code conventions

- follow `.clang-format`, `.editorconfig`, and `docs/en/reference/coding-style.md`
- use `#pragma once`, not include guards
- namespaces: `cuda_academy::`, `tensorcraft::`, `hpc::`, `mini_inference::`
- class names use PascalCase; functions use lower_case; members use trailing `_`
- prefer existing helpers and narrow fixes over new abstractions

## Documentation discipline

- `CHANGELOG.md` in the repository root is the only changelog
- do not duplicate release history inside module docs or GitHub Pages
- keep docs aligned with the current code and build surface
- avoid committing local tool state or generated agent artifacts
