# Copilot Instructions for CUDA Kernel Academy

## Repository intent

This repository is a multi-module CUDA learning project in stabilization mode. Favor targeted fixes, cleanup, documentation clarity, and workflow simplification over new feature growth.

## Repository shape

- `01-sgemm-tutorial/` uses a standalone `Makefile`
- `02-tensorcraft-core/`, `03-hpc-advanced/`, `04-inference-engine/`, `common/`, and `examples/` use root/module CMake
- `04-inference-engine/` depends on `02-tensorcraft-core`
- `03-hpc-advanced/` requires CUDA 12+ and modern compiler support

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

GitHub Actions only runs CPU-safe checks. CUDA runtime validation belongs on a real GPU machine.

## Working rules

- keep changes small and repository-specific
- prefer deleting stale content over adding another abstraction or process layer
- keep GitHub Pages focused on learning and reference material
- keep `CHANGELOG.md` at the repository root as the only changelog
- do not commit local tool state or generated agent artifacts
