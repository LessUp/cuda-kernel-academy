---
name: verify
description: Run the repo-native checks for CUDA Kernel Academy changes.
---

Use this skill to validate changes with the checks that matter for this repository.

## Default sequence

1. Run repo hygiene checks:
   ```bash
   pre-commit run --all-files
   cmake --list-presets
   ```

2. If docs, Pages, or workflow-facing files changed:
   ```bash
   npm ci
   npm run docs:build
   ```

3. If module `01-sgemm-tutorial` changed and a GPU/test environment is available:
   ```bash
   cd 01-sgemm-tutorial && make test
   ```

4. If root/module CMake or CUDA code changed and a GPU environment is available:
   ```bash
   cmake --preset default
   cmake --build --preset default
   ctest --preset default
   ```

## Repository-specific rules

- GitHub-hosted CI is CPU-safe only
- GPU validation must happen on a real CUDA machine
- Prefer the smallest meaningful validation set for the files you changed
